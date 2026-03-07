# WebIdentification

This project can be split into two parts

Part 1: A pipeline for building a dataset from a similar, but not quite what we need, dataset called Mind2Web. It does this through storing data in Neo4j which we can then export as YOLO labels.
Part 2: Training a YOLO model on the generated dataset using Ultralytics.

## Requirements

- Python 3.12+
- Docker (for Neo4j, but can be used with other helpful containers, which limit python needs)
- Optional: Weights & Biases API key for experiment tracking (`WANDB_API_KEY`)

## Local Setup

Install the project and dependencies:

```bash
make install
```

Show available commands:

```bash
$ make help
data_loader     Build the data loader Docker image
database        Start the Neo4j database container
help            Show this help
install         Install the package and its dependencies
load_data       Load data into the database
```

## Locally Running the Pipeline

### 1. Start Neo4j

```bash
make database
```

By default this starts Neo4j at:

- Browser: `http://localhost:7474`
- Bolt: `bolt://localhost:7687`
- Auth: `neo4j/password`

### 2. Load Mind2Web data into Neo4j

```bash
make load_data
```

This builds the container in `containers/data_loader/` and runs `tools/ingest_neo4j.py`.

### 3. Export images + labels (YOLO format)

```bash
python get_data_as_coco.py
```

Optional flags:

- `--zip`: create `CV_WebIdentification.zip`
- `--clean`: delete generated `CV_WebIdentification/` after run

Outputs:

- `CV_WebIdentification/train|test|val/` with `.png` screenshots and `.txt` labels
- `coco8.yaml` with class names and split paths

### 4. Train model

```bash
export WANDB_API_KEY=<your_key>
python main.py
```