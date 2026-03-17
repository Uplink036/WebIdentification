from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
import yaml

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
    label_paths = list((DATA_DIR / split).glob("*.txt"))
    for label_path in tqdm(label_paths, desc=f"Processing {split}", unit="file"):
        overlaps = []
        with open(label_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        filtered_lines = []
        for line in lines:
            invalid = False
            class_id, x_center, y_center, width, height = line.split()
            if (
                float(x_center) < 0.0
                or float(x_center) > 1.0
                or float(y_center) < 0.0
                or float(y_center) > 1.0
                or float(width) < 0.0
                or float(width) > 1.0
                or float(height) < 0.0
                or float(height) > 1.0
            ):
                print(f"Removing out of bounds box from {label_path}")
                invalid = True

            top_left_x = float(x_center) - float(width) / 2
            top_left_y = float(y_center) - float(height) / 2
            if (
                top_left_x < 0.0
                or top_left_x > 1.0
                or top_left_y < 0.0
                or top_left_y > 1.0
            ):
                print(f"Removing out of bounds box from {label_path}")
                invalid = True

            top_right_x = float(x_center) + float(width) / 2
            top_right_y = float(y_center) - float(height) / 2
            if (
                top_right_x < 0.0
                or top_right_x > 1.0
                or top_right_y < 0.0
                or top_right_y > 1.0
            ):
                print(f"Removing out of bounds box from {label_path}")
                invalid = True

            bottom_left_x = float(x_center) - float(width) / 2
            bottom_left_y = float(y_center) + float(height) / 2
            if (
                bottom_left_x < 0.0
                or bottom_left_x > 1.0
                or bottom_left_y < 0.0
                or bottom_left_y > 1.0
            ):
                print(f"Removing out of bounds box from {label_path}")
                invalid = True

            bottom_right_x = float(x_center) + float(width) / 2
            bottom_right_y = float(y_center) + float(height) / 2
            if (
                bottom_right_x < 0.0
                or bottom_right_x > 1.0
                or bottom_right_y < 0.0
                or bottom_right_y > 1.0
            ):
                print(f"Removing out of bounds box from {label_path}")
                invalid = True

            if not invalid:
                filtered_lines.append(line)

        with open(label_path, "w") as f:
            f.write("\n".join(filtered_lines) + "\n")
