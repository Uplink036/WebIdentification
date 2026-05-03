import csv

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")
OUTPUT = "element_tags.csv"

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Element)
            RETURN e.tag AS tag, count(e) AS count
            ORDER BY count DESC
        """)
        rows = result.data()

        with open(OUTPUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tag", "count"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote {len(rows)} rows to {OUTPUT}")
