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
data_pipeline   Run the full data pipeline...
data_loader     Build the data loader Docker image
database        Start the Neo4j database container
fetch_data      Fetch data into ultralytics format as a ZIP file
frontend        Build the frontend Docker image
help            Show this help
install         Install the package and its dependencies
load_data       Load data into the database
model_backend   Build the model backend Docker image
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

### 3. Export images + labels (YOLO format)

```bash
python src/webidentification/pipeline/export_ultralytics_dataset.py
```

Optional flags:

- `--zip`: create `CV_WebIdentification.zip`
- `--clean`: delete generated `CV_WebIdentification/` after run

Outputs:

- `CV_WebIdentification/train|test|val/` with `.png` screenshots and `.txt` labels
- `cv_webidentification.yaml` with class names and split paths

Note, this took about 2 hours to run on my machine.

### 4. Train model

```bash
export WANDB_API_KEY=<your_key>
python src/webidentification/cli/train.py
```

Choose model family:

```bash
python src/webidentification/cli/train.py --model yolo
python src/webidentification/cli/train.py --model rtdetr
```

### 5. Start a W&B sweep

Create a sweep from `docs/sweep.yaml`:

```bash
export WANDB_API_KEY=<your_key>
wandb sweep docs/sweep.yaml --project WebIdentification
```

Start an agent with the returned sweep ID:

```bash
python src/webidentification/cli/sweep.py --sweep-id <your_sweep_id>
```