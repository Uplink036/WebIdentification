.PHONY: install
install: ## Install the package and its dependencies
	pip install -e .[all]

.PHONY: database
database: ## Start the Neo4j database container
	docker run \
	--restart unless-stopped \
	--publish=7474:7474 --publish=7687:7687 \
	--env NEO4J_AUTH=neo4j/password \
	--volume=webidentification_neo4j_data:/data \
	neo4j:2026.01.4

.PHONY: data_loader
data_loader: ## Build the data loader Docker image
	docker build -f ./containers/data_loader/Dockerfile -t webidentification_data_loader:latest .

.PHONY: load_data
load_data: data_loader ## Load data into the database
	./tools/docker_ingest.sh

.PHONY: fetch_data
fetch_data: ## Fetch data into ultralytics format as a ZIP file 
	python3 tools/get_data_as_coco.py

.PHONY: data_pipeline
data_pipeline: load_data fetch_data ## Run the full data pipeline
	@echo "Running the full data pipeline..."

.PHONY: help 
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'