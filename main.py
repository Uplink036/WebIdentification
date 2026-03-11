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
SPLIT_ULTRALYTICS_NAME = Path("ultralytics_split.yaml")
SPLIT_TRAIN_DIR_NAME = "split_train"
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


def load_ultralytics_config(config_path: Path) -> dict:
    """Load the base Ultralytics dataset config YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_split_config(base_config: dict, split_config_path: Path, split_train_dir_name: str) -> None:
    """Write a temporary Ultralytics config that points train to split_train."""
    split_config = deepcopy(base_config)
    split_config["train"] = split_train_dir_name
    with split_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(split_config, file)


def iter_split_batches(image_files: list[Path], split_points: list[int]):
    """Yield incremental image batches between consecutive split points."""
    for previous_split, current_split in zip(split_points, split_points[1:]):
        yield previous_split, current_split, image_files[previous_split:current_split]


def reset_split_train_dir(split_train_dir: Path) -> None:
    """Recreate split train directory from scratch."""
    if split_train_dir.exists():
        shutil.rmtree(split_train_dir)
    split_train_dir.mkdir(parents=True, exist_ok=True)


def copy_split_files(image_files: list[Path], split_train_dir: Path) -> None:
    """Copy image and annotation pairs into split train directory."""
    for image_file in image_files:
        shutil.copy(image_file, split_train_dir / image_file.name)
        annotation_file = image_file.with_suffix(".txt")
        shutil.copy(annotation_file, split_train_dir / annotation_file.name)


def main() -> None:
    load_dotenv()
    settings.update({"wandb": True})
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    ultralytics_config = load_ultralytics_config(CONFIG_PATH)
    data_dir = Path(ultralytics_config["path"])
    train_dir = data_dir / ultralytics_config["train"]
    split_train_dir = data_dir / SPLIT_TRAIN_DIR_NAME

    write_split_config(ultralytics_config, SPLIT_ULTRALYTICS_NAME, SPLIT_TRAIN_DIR_NAME)
    reset_split_train_dir(split_train_dir)

    all_train_images = sorted(train_dir.glob("*.png"))
    splits = [int(len(all_train_images) * percentage) for percentage in PERCENTAGE_TRAIN_SPLITS]

    train_device = 0 if torch.cuda.is_available() else "cpu"
    train_workers = pick_dataloader_workers()
    batch_size = BATCH_UTILIZATION_TARGET

    for previous_split, current_split, new_files in iter_split_batches(all_train_images, splits):
        copy_split_files(new_files, split_train_dir)
        model = YOLO(MODEL)
        model.train(
            data=str(SPLIT_ULTRALYTICS_NAME),
            epochs=100,
            project="WebIdentification",
            name=f"yolo26n_split_{previous_split}_{current_split}-0",
            batch=batch_size,
            imgsz=640,
            workers=train_workers,
            device=train_device,
            patience=10,
        )


if __name__ == "__main__":
    main()
