import hashlib
from math import ceil
from tqdm import tqdm
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Action)
            WHERE t.screenshot_b64 IS NOT NULL
            RETURN count(t) AS count
        """)
        count_actions = result.single()["count"]
        unique_hashes = set()

        result = session.run("""
            MATCH (t:Action)
            WHERE t.screenshot_b64 IS NOT NULL
            RETURN t.screenshot_b64 AS screenshot
        """)

        batch_size = 100
        batch_iterator = iter(lambda: result.fetch(batch_size), [])
        for rows in tqdm(batch_iterator, total=ceil(count_actions/batch_size), unit="row", desc=f"{batch_iterator}x rows checked and hashed"):
            for row in rows:
                screenshot = row["screenshot"]
                screenshot_hash = hashlib.md5(screenshot.encode()).hexdigest()
                unique_hashes.add(screenshot_hash)

        count_actions_distinct = len(unique_hashes)
        count_actions_duplicate = count_actions - count_actions_distinct

        print(f"{'Count Actions':<50} {count_actions:>10}")
        print(f"{'Count Distinct Actions':<50} {count_actions_distinct:>10}")
        print(f"{'Count Duplicate Actions':<50} {count_actions_duplicate:>10}")
