import os
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import wandb
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO, settings

MODEL = "yolo26n.pt"
CONFIG_PATH = Path("cv_webidentification.yaml")
PERCENTAGE_TRAIN_SPLITS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_WORKERS = 4
AUTO_BATCH_SIZE = True
BATCH_UTILIZATION_TARGET = -1 if AUTO_BATCH_SIZE else 0.8

def get_available_shm_gb() -> float:
    """Return available /dev/shm size in GB; 0 when unavailable."""
    try:
        shm_stats = os.statvfs("/dev/shm")
        return (shm_stats.f_bavail * shm_stats.f_frsize) / (1024 ** 3)
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
    load_dotenv()
    settings.update({"wandb": True})
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    train_device = 0 if torch.cuda.is_available() else "cpu"
    train_workers = pick_dataloader_workers()
    batch_size = BATCH_UTILIZATION_TARGET

    for split in PERCENTAGE_TRAIN_SPLITS:
        model = YOLO(MODEL)
        model.train(
            data=str(CONFIG_PATH),
            epochs=100,
            project="WebIdentification",
            name=f"{MODEL}_split_{split}-0",
            batch=batch_size,
            imgsz=640,
            workers=train_workers,
            device=train_device,
            patience=10,
            fraction=split,
        )


if __name__ == "__main__":
    main()
