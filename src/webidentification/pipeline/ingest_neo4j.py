import base64
import json
import os
import signal
import sys
import time
from io import BytesIO

from datasets import load_dataset
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from tqdm import tqdm

URI = os.getenv("URI", "bolt://localhost:7687")
AUTH = (os.getenv("USERNAME", "neo4j"), os.getenv("PASSWORD", "password"))

SPLITS = ["train", "test_domain", "test_task", "test_website"]

shutdown_requested = False


def handle_shutdown(signum, frame):
    """Handle SIGTERM signal from Kubernetes or SIGINT from Ctrl+C"""
    global shutdown_requested
    print(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


def create_indexes(session):
    """Create indexes for faster MERGE operations"""
    session.run("CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.id)")
    session.run("CREATE INDEX IF NOT EXISTS FOR (a:Action) ON (a.id)")
    session.run("CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.key)")


def send_row_to_database(session, information):
    t0 = time.time()

    session.run(
        """
    MERGE (t:Task {id: $annotation_id})
    SET t.description = $task, t.website = $website, t.domain = $domain, t.subdomain = $subdomain
    MERGE (a:Action {id: $action_uid})
    SET a.op = $op, a.value = $value, a.raw_html = $raw_html, a.cleaned_html = $cleaned_html, a.screenshot_b64 = $screenshot_b64, a.type = $split_type
    MERGE (t)-[:HAS_ACTION]->(a)
    WITH a
    UNWIND $pos_cands AS elem
    MERGE (e:Element {key: elem.key})
    SET e.backend_node_id = elem.backend_node_id, e.tag = elem.tag, e.is_target = elem.is_target, e.attributes = elem.attributes
    MERGE (a)-[:TARGETS]->(e)
    WITH a
    UNWIND $neg_cands AS elem
    MERGE (e:Element {key: elem.key})
    SET e.backend_node_id = elem.backend_node_id, e.tag = elem.tag, e.is_target = elem.is_target, e.attributes = elem.attributes
    MERGE (a)-[:HAS_CANDIDATE]->(e)
    """,
        **information,
    )

    return time.time() - t0


def encode_screenshot(img):
    """Optimization 2: Cleaner screenshot encoding"""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_positive_elements(row, pos_cands):
    processed_pos = []
    for elem in pos_cands:
        elem = json.loads(elem)
        key = f"{row['action_uid']}::{elem['backend_node_id']}"
        processed_pos.append({**elem, "key": key})

    return processed_pos


def extract_negative_elements(row, neg_cands):
    processed_neg = []
    for elem in neg_cands:
        elem = json.loads(elem)
        key = f"{row['action_uid']}::{elem['backend_node_id']}"
        processed_neg.append({**elem, "key": key})

    return processed_neg


def exists_in_db(session, action_uid):
    result = session.run(
        "MATCH (a:Action {id: $action_uid}) RETURN count(a) > 0 AS exists",
        action_uid=action_uid,
    )
    return result.single()["exists"]


def verify_database_connection(driver):
    try:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {URI} as {AUTH[0]=}.")
    except (ServiceUnavailable, AuthError):
        print("ERROR: Could not connect to Neo4j before starting ingestion.")
        sys.exit(1)
    except Exception:
        print("ERROR: Unexpected error while connecting to Neo4j.")
        sys.exit(1)


with GraphDatabase.driver(URI, auth=AUTH) as driver:
    verify_database_connection(driver)
    with driver.session() as session:
        print("Creating indexes...")
        create_indexes(session)

        total_pos = 0
        total_neg = 0
        total_db_time = 0
        total_rows = 0

        for split in SPLITS:
            if shutdown_requested:
                print("Shutdown requested, stopping ingestion...")
                break

            print(f"\nProcessing split: {split}")

            dataset = load_dataset(
                "osunlp/Multimodal-Mind2Web", split=split, streaming=True
            )

            split_pos = 0
            split_neg = 0
            split_rows = 0

            for row in tqdm(dataset):
                if shutdown_requested:
                    print("Shutdown requested, stopping ingestion...")
                    break

                if exists_in_db(session, row["action_uid"]):
                    continue

                if row["screenshot"] is None:
                    print(
                        f"Skipping annotation {row['annotation_id']} due to missing screenshot"
                    )
                    continue

                op = json.loads(row["operation"])
                pos_cands = row["pos_candidates"]
                neg_cands = row["neg_candidates"]

                processed_pos = extract_positive_elements(row, pos_cands)
                processed_neg = extract_negative_elements(row, neg_cands)

                split_pos += len(processed_pos)
                split_neg += len(processed_neg)
                split_rows += 1

                information = {
                    "annotation_id": row["annotation_id"],
                    "task": row["confirmed_task"],
                    "website": row["website"],
                    "domain": row["domain"],
                    "subdomain": row["subdomain"],
                    "action_uid": row["action_uid"],
                    "op": op["op"],
                    "value": op.get("value", ""),
                    "raw_html": row["raw_html"],
                    "cleaned_html": row["cleaned_html"],
                    "screenshot_b64": encode_screenshot(row["screenshot"]),
                    "split_type": split,
                    "pos_cands": processed_pos,
                    "neg_cands": processed_neg,
                }

                db_time = send_row_to_database(session, information)
                total_db_time += db_time

            total_pos += split_pos
            total_neg += split_neg
            total_rows += split_rows

            print(f"  Rows: {split_rows}")
            print(
                f"  Positive candidates: {split_pos} (avg: {split_pos/split_rows:.1f} per row)"
            )
            print(
                f"  Negative candidates: {split_neg} (avg: {split_neg/split_rows:.1f} per row)"
            )

        print(f"\n=== TOTALS ===")
        print(f"Total rows: {total_rows}")
        print(f"Total positive candidates: {total_pos}")
        print(f"Total negative candidates: {total_neg}")
        print(f"Total DB time: {total_db_time:.2f}s")
        print(f"Avg DB time per row: {total_db_time/(total_rows):.3f}s")

print("Done.")
