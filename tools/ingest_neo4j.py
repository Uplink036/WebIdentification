from datasets import load_dataset
from neo4j import GraphDatabase
import json
import base64
from tqdm import tqdm
from io import BytesIO

URI = "bolt://database:7687"
AUTH = ("neo4j", "password")

SPLITS = ["train", "test_domain", "test_task", "test_website"]
BATCH_SIZE = 10


def flush_batch(session, batch):
    if not batch:
        return

    session.run("""
    UNWIND $rows AS row

    MERGE (t:Task {id: row.annotation_id})
    SET t.description = row.task,
        t.website = row.website,
        t.domain = row.domain,
        t.subdomain = row.subdomain

    MERGE (a:Action {id: row.action_uid})
    SET a.op = row.op,
        a.value = row.value,
        a.raw_html = row.raw_html,
        a.cleaned_html = row.cleaned_html,
        a.screenshot_b64 = row.screenshot_b64,
        a.type = row.split_type

    MERGE (t)-[:HAS_ACTION]->(a)

    WITH a, row

    UNWIND row.pos_cands AS pos
        MERGE (e:Element {key: pos.key})
        SET e.backend_node_id = pos.backend_node_id,
            e.tag = pos.tag,
            e.is_target = pos.is_original_target,
            e.attributes = pos.attributes
        MERGE (a)-[:TARGETS]->(e)

    WITH a, row

    UNWIND row.neg_cands AS neg
        MERGE (e:Element {key: neg.key})
        SET e.backend_node_id = neg.backend_node_id,
            e.tag = neg.tag,
            e.is_target = false,
            e.attributes = neg.attributes
        MERGE (a)-[:HAS_CANDIDATE]->(e)
    """, rows=batch)


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

        for split in SPLITS:
            print(f"\nProcessing split: {split}")

            dataset = load_dataset(
                "osunlp/Multimodal-Mind2Web",
                split=split,
                streaming=True
            )

            batch = []

            for row in tqdm(dataset):

                if row["screenshot"] is None:
                    continue

                op = json.loads(row["operation"])
                pos_cands = row["pos_candidates"]
                neg_cands = row["neg_candidates"]

                processed_pos = extract_positive_elements(row, pos_cands)
                processed_neg = extract_negative_elements(row, neg_cands)

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
                    "screenshot_b64": (lambda img: (buf := BytesIO(), img.save(buf, format='PNG'), base64.b64encode(buf.getvalue()).decode("utf-8"))[2])(row["screenshot"]),
                    "split_type": split,
                    "pos_cands": processed_pos,
                    "neg_cands": processed_neg,
                })

                if len(batch) >= BATCH_SIZE:
                    flush_batch(session, batch)
                    batch = []

            # Flush remaining
            flush_batch(session, batch)

print("Done.")