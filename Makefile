install:
	pip install -e .[all]

database:
	docker run \
    --restart unless-stopped \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/password \
    --volume=./data:/data \
    neo4j:2026.01.4

data_loader:
	docker build -f ./containers/data_loader/Dockerfile -t webidentification_data_loader:latest .

load_data: data_loader
	./tools/docker_ingest.sh