from copy import deepcopy
import os
import shutil
import wandb
import torch
from dotenv import load_dotenv

def get_available_shm_gb() -> float:
    """Return available /dev/shm size in GB; 0 when unavailable."""
    try:
        shm_stats = os.statvfs("/dev/shm")
        return (shm_stats.f_bavail * shm_stats.f_frsize) / (1024 ** 3)
    except OSError:
        return 0.0

from ultralytics import settings, YOLO
import yaml
from pathlib import Path

def pick_dataloader_workers(default_workers: int = 4) -> int:
    """Choose workers conservatively when shared memory is constrained."""
    shm_gb = get_available_shm_gb()
    if shm_gb < 1.0:
        return 0
    if shm_gb < 2.0:
        return 1

    cpu_count = os.cpu_count() or 1
    return max(1, min(default_workers, cpu_count // 2))
load_dotenv()
settings.update({"wandb": True})
TRAIN_WORKERS = pick_dataloader_workers(default_workers=4)

# WANDB login
wandb.login(key=os.environ.get("WANDB_API_KEY"))

with open("cv_webidentification.yaml", "r") as f:
    ultralytics_config = yaml.safe_load(f)

MODEL = "yolo26n.pt"
model = YOLO(MODEL)


DATA_DIR = Path(ultralytics_config["path"])
TRAIN_DIR = DATA_DIR / ultralytics_config["train"]
VAL_DIR = DATA_DIR / ultralytics_config["val"]
TEST_DIR = DATA_DIR / ultralytics_config["test"]

SPLIT_ULTRALYTICS_NAME = "ultralytics_split.yaml"
SPLIT_ULTRALYTICS = deepcopy(ultralytics_config)
SPLIT_TRAIN_DIR_NAME = "split_train"
SPLIT_TRAIN_DIR = DATA_DIR / SPLIT_TRAIN_DIR_NAME

SPLIT_ULTRALYTICS["train"] = SPLIT_TRAIN_DIR_NAME
yaml.dump(SPLIT_ULTRALYTICS, open(SPLIT_ULTRALYTICS_NAME, "w"))

PERCENTAGE_SPLITS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TRAIN_DEVICE = 0 if torch.cuda.is_available() else "cpu"

def get_available_memory_gb(device: int | str) -> float:
    """Return available memory in GB for the selected training device."""
    if device == "cpu":
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return (page_size * available_pages) / (1024 ** 3)
    else:
        free_mem_bytes, _ = torch.cuda.mem_get_info(device)
        return free_mem_bytes / (1024 ** 3)


def pick_batch_size(device: int | str, mem_per_item_gb: float = 1.1, max_batch: int = 16) -> int:
    """Choose batch size from available memory with safety bounds."""
    available_mem_gb = get_available_memory_gb(device)
    return max(1, min(max_batch, int(available_mem_gb // mem_per_item_gb)))


BATCH_SIZE = pick_batch_size(TRAIN_DEVICE, mem_per_item_gb=1.1, max_batch=16)

def iter_split_batches(image_files: list[Path], split_points: list[int]):
    """Yield incremental image batches between consecutive split points."""
    for previous_split, current_split in zip(split_points, split_points[1:]):
        yield previous_split, current_split, image_files[previous_split:current_split]

# Remove existing split train directory if it exists
if SPLIT_TRAIN_DIR.exists():
    shutil.rmtree(SPLIT_TRAIN_DIR)
SPLIT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
all_train_images = sorted((TRAIN_DIR).glob("*.png"))
# Based on percentage split, make a splits 
splits = [int(len(all_train_images) * p) for p in PERCENTAGE_SPLITS]
for previous_split, current_split, new_files in iter_split_batches(all_train_images, splits):
    for file in new_files:
        shutil.copy(file, SPLIT_TRAIN_DIR / file.name)
        annotation_file = file.with_suffix(".txt")
        shutil.copy(annotation_file, SPLIT_TRAIN_DIR / annotation_file.name)

    model.train(
        data=SPLIT_ULTRALYTICS_NAME,
        epochs=100,
        project="ultralytics",
        name=f"yolo26n_split_{previous_split}_{current_split}-0",
        batch=BATCH_SIZE,
        imgsz=640,
        workers=TRAIN_WORKERS,
        device=TRAIN_DEVICE,
    )
