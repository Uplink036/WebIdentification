from datasets import load_dataset
from neo4j import GraphDatabase
import json
import base64
from tqdm import tqdm
from io import BytesIO
import time

URI = "bolt://database:7687"
AUTH = ("neo4j", "password")

SPLITS = ["train", "test_domain", "test_task", "test_website"]
BATCH_SIZE = 5


def create_indexes(session):
    """Create indexes for faster MERGE operations"""
    session.run("CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.id)")
    session.run("CREATE INDEX IF NOT EXISTS FOR (a:Action) ON (a.id)")
    session.run("CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.key)")


def flush_batch(session, batch):
    if not batch:
        return

    t0 = time.time()
    
    # Single query with all operations to avoid transaction conflicts
    session.run("""
    UNWIND $rows AS row
    MERGE (t:Task {id: row.annotation_id})
    SET t.description = row.task, t.website = row.website, t.domain = row.domain, t.subdomain = row.subdomain
    MERGE (a:Action {id: row.action_uid})
    SET a.op = row.op, a.value = row.value, a.raw_html = row.raw_html, a.cleaned_html = row.cleaned_html, a.screenshot_b64 = row.screenshot_b64, a.type = row.split_type
    MERGE (t)-[:HAS_ACTION]->(a)
    WITH a, row
    UNWIND row.pos_cands AS elem
    MERGE (e:Element {key: elem.key})
    SET e.backend_node_id = elem.backend_node_id, e.tag = elem.tag, e.is_target = elem.is_target, e.attributes = elem.attributes
    MERGE (a)-[:TARGETS]->(e)
    WITH a, row
    UNWIND row.neg_cands AS elem
    MERGE (e:Element {key: elem.key})
    SET e.backend_node_id = elem.backend_node_id, e.tag = elem.tag, e.is_target = elem.is_target, e.attributes = elem.attributes
    MERGE (a)-[:HAS_CANDIDATE]->(e)
    """, rows=batch)
    
    return time.time() - t0


def encode_screenshot(img):
    """Optimization 2: Cleaner screenshot encoding"""
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_positive_elements(row, pos_cands):
    processed_pos = []
    for elem in pos_cands:
        elem = json.loads(elem)
        key = f"{row['action_uid']}::{elem['backend_node_id']}"
        processed_pos.append({
                        **elem,
                        "key": key
                    })
        
    return processed_pos

def extract_negative_elements(row, neg_cands):
    processed_neg = []
    for elem in neg_cands:
        elem = json.loads(elem)
        key = f"{row['action_uid']}::{elem['backend_node_id']}"
        processed_neg.append({
                        **elem,
                        "key": key
                    })
        
    return processed_neg

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:

        # Optimization 1: Create indexes before ingestion
        print("Creating indexes...")
        create_indexes(session)

        total_pos = 0
        total_neg = 0
        total_db_time = 0
        total_rows = 0

        for split in SPLITS:
            print(f"\nProcessing split: {split}")

            dataset = load_dataset(
                "osunlp/Multimodal-Mind2Web",
                split=split,
                streaming=True
            )

            batch = []
            split_pos = 0
            split_neg = 0
            split_rows = 0

            for row in tqdm(dataset):

                if row["screenshot"] is None:
                    continue

                op = json.loads(row["operation"])
                pos_cands = row["pos_candidates"]
                neg_cands = row["neg_candidates"]

                processed_pos = extract_positive_elements(row, pos_cands)
                processed_neg = extract_negative_elements(row, neg_cands)

                split_pos += len(processed_pos)
                split_neg += len(processed_neg)
                split_rows += 1

                batch.append({
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
                })

                if len(batch) >= BATCH_SIZE:
                    db_time = flush_batch(session, batch)
                    total_db_time += db_time
                    batch = []

            # Flush remaining
            db_time = flush_batch(session, batch)
            if db_time:
                total_db_time += db_time

            total_pos += split_pos
            total_neg += split_neg
            total_rows += split_rows

            print(f"  Rows: {split_rows}")
            print(f"  Positive candidates: {split_pos} (avg: {split_pos/split_rows:.1f} per row)")
            print(f"  Negative candidates: {split_neg} (avg: {split_neg/split_rows:.1f} per row)")

        print(f"\n=== TOTALS ===")
        print(f"Total rows: {total_rows}")
        print(f"Total positive candidates: {total_pos}")
        print(f"Total negative candidates: {total_neg}")
        print(f"Total DB time: {total_db_time:.2f}s")
        print(f"Avg DB time per batch: {total_db_time/(total_rows/BATCH_SIZE):.3f}s")

print("Done.")
