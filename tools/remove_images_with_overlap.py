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
        bboxs = [
            BBox(
                int(class_id),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
            )
            for line in lines
            for class_id, x_center, y_center, width, height in [line.split()]
        ]
        for index, outer_bbox in enumerate(bboxs):
            outer_left = outer_bbox.x_center - outer_bbox.width / 2
            outer_right = outer_bbox.x_center + outer_bbox.width / 2
            outer_top = outer_bbox.y_center - outer_bbox.height / 2
            outer_bottom = outer_bbox.y_center + outer_bbox.height / 2
            for inner_bbox in bboxs[index + 1 :]:
                inner_left = inner_bbox.x_center - inner_bbox.width / 2
                inner_right = inner_bbox.x_center + inner_bbox.width / 2
                inner_top = inner_bbox.y_center - inner_bbox.height / 2
                inner_bottom = inner_bbox.y_center + inner_bbox.height / 2

                overlap_x = max(
                    0, min(outer_right, inner_right) - max(outer_left, inner_left)
                )
                overlap_y = max(
                    0, min(outer_bottom, inner_bottom) - max(outer_top, inner_top)
                )
                overlap_area = overlap_x * overlap_y
                overlaps.append(overlap_area)
        if any(overlap > 0.05 for overlap in overlaps):
            print(f"Removing {label_path} due to bbox overlap")
            os.remove(label_path)
            png_path = label_path.with_suffix(".png")
            if png_path.exists():
                os.remove(png_path)
