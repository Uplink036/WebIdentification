from neo4j import GraphDatabase

URI = "bolt://database:7687"
AUTH = ("neo4j", "password")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        session.run("MATCH ()-[r]-() DELETE r")
        while True:
            result = session.run(
                """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                WITH n,r LIMIT 10000
                DELETE n,r
                RETURN count(n) as deletedNodesCount
            """
            )
            count = result.single()["deletedNodesCount"]
            print(f"Deleted {count} nodes")
            if count == 0:
                break
        print("Database cleaned")
