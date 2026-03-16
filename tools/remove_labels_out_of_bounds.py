import os
from pathlib import Path
import yaml
import tqdm

DATA_CONFIGURATION = yaml.safe_load(
    Path("/workspaces/WebIdentification/cv_webidentification.yaml").read_text()
)
ROOT_DIR = Path("/workspaces/WebIdentification")
DATA_DIR = ROOT_DIR / DATA_CONFIGURATION["path"]
SPLITS = ["train", "val", "test"]

@dataclass
class BBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


for split in SPLITS:
    for label_path in tqdm(list((DATA_DIR / split).glob("*.txt")), desc=f"Processing {split}", unit="file"):
        overlaps = []
        with open(label_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            class_id, x_center, y_center, width, height = line.split()
            if float(x_center) < 0.0 or float(x_center) > 1.0 or float(y_center) < 0.0 or float(y_center) > 1.0 or float(width) < 0.0 or float(width) > 1.0 or float(height) < 0.0 or float(height) > 1.0:
                print(f"Removing out of bounds box from {label_path}")
                lines.remove(line)

            top_left_x = float(x_center) - float(width) / 2
            top_left_y = float(y_center) - float(height) / 2
            if top_left_x < 0.0 or top_left_x > 1.0 or top_left_y < 0.0 or top_left_y > 1.0:
                print(f"Removing out of bounds box from {label_path}")
                lines.remove(line)

            top_right_x = float(x_center) + float(width) / 2
            top_right_y = float(y_center) - float(height) / 2
            if top_right_x < 0.0 or top_right_x > 1.0 or top_right_y < 0.0 or top_right_y > 1.0:
                print(f"Removing out of bounds box from {label_path}")
                lines.remove(line)

            bottom_left_x = float(x_center) - float(width) / 2
            bottom_left_y = float(y_center) + float(height) / 2
            if bottom_left_x < 0.0 or bottom_left_x > 1.0 or bottom_left_y < 0.0 or bottom_left_y > 1.0:
                print(f"Removing out of bounds box from {label_path}")
                lines.remove(line)

            bottom_right_x = float(x_center) + float(width) / 2
            bottom_right_y = float(y_center) + float(height) / 2
            if bottom_right_x < 0.0 or bottom_right_x > 1.0 or bottom_right_y < 0.0 or bottom_right_y > 1.0:
                print(f"Removing out of bounds box from {label_path}")
                lines.remove(line)