import argparse
import os
from pathlib import Path
from random import random
import pandas as pd
import yaml

from tqdm import tqdm
import torch
from dotenv import load_dotenv
from sklearn.model_selection import KFold
from ultralytics import RTDETR, YOLO, settings
import shutil

import wandb

MODEL_WEIGHTS = {"yolo": "yolo26n.pt", "rtdetr": "rtdetr-l.pt"}
CONFIG_PATH = Path("ultralytics.yaml")
PERCENTAGE_TRAIN_SPLITS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_WORKERS = 4
AUTO_BATCH_SIZE = True
BATCH_UTILIZATION_TARGET = -1 if AUTO_BATCH_SIZE else 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WebIdentification model")
    parser.add_argument(
        "--model",
        choices=("yolo", "rtdetr"),
        default="yolo",
        help="Choose yolo or rtdetr",
    )
    return parser.parse_args()


def get_available_shm_gb() -> float:
    """Return available /dev/shm size in GB; 0 when unavailable."""
    try:
        shm_stats = os.statvfs("/dev/shm")
        return (shm_stats.f_bavail * shm_stats.f_frsize) / (1024**3)
    except OSError:
        return 0.0


def pick_dataloader_workers() -> int:
    """Choose workers conservatively when shared memory is constrained."""
    shm_gb = get_available_shm_gb()
    if shm_gb < 1.0:
        return 0
    if shm_gb < 2.0:
        return 1

    cpu_count = os.cpu_count() or 1
    return max(1, min(MAX_WORKERS, cpu_count // 2))


def main() -> None:
    args = parse_args()
    load_dotenv()
    settings.update({"wandb": True})
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    yaml_config = CONFIG_PATH.read_text()
    config = yaml.safe_load(yaml_config)
    data_path = Path(config["path"])
    train_path = data_path / config["train"]
    labels = sorted(train_path.rglob("*.txt"))
    classes = sorted(config["names"].keys())
    index = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=classes, index=index)


    from collections import Counter

    for label in labels:
        lbl_counter = Counter()

        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ", 1)[0])] += 1

        labels_df.loc[label.stem] = lbl_counter
    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

    print(labels_df.head())

    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results
    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df.loc[labels_df.iloc[train].index, f"split_{i}"] = "train"
        folds_df.loc[labels_df.iloc[val].index, f"split_{i}"] = "val"

    ds_yamls = []

    images = sorted(train_path.rglob("*.png"))
    save_path = Path("k_fold_splits")
    save_path.mkdir(parents=True, exist_ok=True)
    for split in folds_df.columns:
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )

    for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
        for split, k_split in folds_df.loc[image.stem].items():
            if k_split not in ("train", "val"):
                print(f"Skipping {image.stem} for {split} as it is marked {k_split}")
                continue
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)

    folds_df.to_csv(save_path / "kfold_datasplit.csv")

    model_key = args.model
    model_weights = MODEL_WEIGHTS[model_key]
    train_device = 0 if torch.cuda.is_available() else "cpu"
    train_workers = pick_dataloader_workers()
    batch_size = BATCH_UTILIZATION_TARGET
    for k, dataset_yaml in enumerate(ds_yamls):
        model = RTDETR(model_weights) if model_key == "rtdetr" else YOLO(model_weights)
        model.train(
            data=dataset_yaml,
            epochs=100,
            project="WebIdentification_k_fold",
            name=f"{model_key}_split_{k}-0",
            batch=batch_size,
            imgsz=640,
            workers=train_workers,
            device=train_device,
            patience=10,
        )


if __name__ == "__main__":
    main()
