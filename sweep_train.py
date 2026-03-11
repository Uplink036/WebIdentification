import os
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from ultralytics import RTDETR, YOLO, settings

CONFIG_PATH = Path("cv_webidentification.yaml")
MAX_WORKERS = 4
AUTO_BATCH_SIZE = True
BATCH_UTILIZATION_TARGET = -1 if AUTO_BATCH_SIZE else 0.8


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


def run_sweep_train() -> None:
    """W&B agent callback: run one training job from sweep config."""
    load_dotenv()
    settings.update({"wandb": True})

    with wandb.init():
        model_weights = wandb.config.get("model", "yolo26n.pt")
        split = float(wandb.config.get("fraction", 1.0))
        epochs = int(wandb.config.get("epochs", 100))
        imgsz = int(wandb.config.get("imgsz", 640))
        patience = int(wandb.config.get("patience", 10))

        train_device = 0 if torch.cuda.is_available() else "cpu"
        train_workers = pick_dataloader_workers()
        batch_size = BATCH_UTILIZATION_TARGET
        if model_weights == "yolo26n.pt":
            model = YOLO(model_weights)
        elif model_weights == "rtdetr-l.pt":
            model = RTDETR(model_weights)

        model.train(
            data=str(CONFIG_PATH),
            epochs=epochs,
            project="WebIdentification",
            name=f"{model_weights}_{Path(model_weights).stem}_split_{split}-0",
            batch=batch_size,
            imgsz=imgsz,
            workers=train_workers,
            device=train_device,
            patience=patience,
            fraction=split,
        )


if __name__ == "__main__":
    run_sweep_train()
