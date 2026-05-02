from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Task)
            RETURN t.website AS website, count(t) AS count
            ORDER BY count DESC
        """)
        records = result.data()

        total_result = session.run("""
            MATCH (t:Task)
            RETURN count(DISTINCT t.website) AS totalWebsites, count(t) AS totalTasks
        """)
        totals = total_result.single()

        print(f"{'Domain':<50} {'Task Count':>10}")
        for record in records:
            print(f"{record['website']:<50} {record['count']:>10}")
        print(f"Total distinct domains: {totals['totalWebsites']}")
        print(f"Total tasks: {totals['totalTasks']}")
