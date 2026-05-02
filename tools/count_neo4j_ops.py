from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Action)
            RETURN a.op AS op, count(a) AS count
            ORDER BY count DESC
        """)
        records = result.data()

        print(f"{'Operation':<20} {'Count':>10}")
        total = 0
        for record in records:
            count = record['count']
            total += count
            print(f"{record['op']:<20} {count:>10}")
        print(f"{'Total':<20} {total:>10}")
        
