import hashlib
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

def hash_label_line(line: str) -> str:
    return hashlib.blake2b(line.encode("utf-8"), digest_size=16).hexdigest()
    

for split in SPLITS:
    label_paths = list((DATA_DIR / split).glob("*.txt"))
    lines_removed = 0
    for label_path in label_paths:
        with open(label_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        hashed_lines = set(lines)
        if len(hashed_lines) != len(lines):
            print(f"Removing {label_path} due to duplicate lines")
            os.remove(label_path)   
            with open(label_path, "w") as f:
                f.write("\n".join(hashed_lines) + "\n") 
            lines_removed += 1
    print(f"Removed {lines_removed} files with duplicate lines in {split} split")