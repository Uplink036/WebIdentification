import os
from pathlib import Path

import yaml
from tqdm import tqdm

DATA_CONFIGURATION = yaml.safe_load(
    Path("/workspaces/WebIdentification/cv_webidentification.yaml").read_text()
)
ROOT_DIR = Path("/workspaces/WebIdentification")
DATA_DIR = ROOT_DIR / DATA_CONFIGURATION["path"]
SPLITS = ["train", "val", "test"]


for split in SPLITS:
    label_paths = list((DATA_DIR / split).glob("*.txt"))
    for label_path in tqdm(label_paths, desc=f"Processing {split}", unit="file"):
        with open(label_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        lines_to_remove = []
        for line in lines:
            class_id, x_center, y_center, width, height = line.split()
            if (
                float(x_center) == 0.0
                and float(y_center) == 0.0
                and float(width) == 0.0
                and float(height) == 0.0
            ):
                print(f"Removing corner box from {label_path}")
                lines_to_remove.append(line)
        for line in lines_to_remove:
            lines.remove(line)
        if len(lines) >= 1:
            with open(label_path, "w") as f:
                f.write("\n".join(lines) + "\n")
        else:
            print(f"No valid boxes left in {label_path}, removing file")
            os.remove(label_path)
            png_path = label_path.with_suffix(".png")
            if png_path.exists():
                os.remove(png_path)
