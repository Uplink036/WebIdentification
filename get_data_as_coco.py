#!/usr/bin/env python3
"""Tool to retrieve screenshots from Mind2Web database and save to COCO format."""

import base64
import io
import pathlib
import shutil
import sys
import json
import yaml
from neo4j import GraphDatabase
from PIL import Image
from tqdm import tqdm

URI = "bolt://database:7687"
AUTH = ("neo4j", "password")

ROOT_DIR = pathlib.Path("./CV_WebIdentification")
TRAIN_DIR = ROOT_DIR / "train"
TEST_DIR = ROOT_DIR / "test"
VAL_DIR = ROOT_DIR / "val"

# Configure which element tags to include and how to group them
ELEMENT_FILTER = {
    "button": "button",
    "input": "input",
    "a": "link",
    "select": "select",
    "textarea": "textarea",
    # Add more mappings as needed: "tag_name": "class_name"
}

def get_safe_filename(action_uid, format="png"):
    filename = f"screenshot_{action_uid}.{format}"
    return filename

def save_screenshot(action_uid, screenshot_b64, dir):
    img_data = base64.b64decode(screenshot_b64)
    img = Image.open(io.BytesIO(img_data))
    filename = get_safe_filename(action_uid, "png")
    img.save(dir / filename)
    return img.size

def save_bbox(action_uid, elements, img_width, img_height, dir, class_names):
    label_file = get_safe_filename(action_uid, "txt")
    with open(dir / label_file, "w") as f:
        for elem in elements:
            if elem["tag"] not in ELEMENT_FILTER:
                continue
            attrs = json.loads(elem["attributes"])
            bbox_str = attrs.get("bounding_box_rect")
            if not bbox_str:
                continue
            x_center, y_center, width, height = map(float, bbox_str.split(","))
            class_name = ELEMENT_FILTER[elem["tag"]]
            class_id = class_names.index(class_name)
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=AUTH)
    data = {"path": "CV_WebIdentification", "train": "train", "test": "test", "val": "val"}
    
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with driver.session() as session:
            data["names"] = {i: name for i, name in enumerate(set(ELEMENT_FILTER.values()))}
            class_names = list(data["names"].values())

        with driver.session() as session:
            results = session.run("""
                MATCH (a:Action)
                OPTIONAL MATCH (a)-[:TARGETS]->(target:Element)
                OPTIONAL MATCH (a)-[:HAS_CANDIDATE]->(candidate:Element)
                RETURN a.id AS action_uid, a.screenshot_b64 AS screenshot, 
                       a.type AS split, collect(DISTINCT target) + collect(DISTINCT candidate) AS elements
            """)
            
            for record in tqdm(results):
                split = record["split"]
                current_dir = VAL_DIR if split == "test_website" else (TEST_DIR if "test" in split else TRAIN_DIR)
                
                img_width, img_height = save_screenshot(record["action_uid"], record["screenshot"], current_dir)
                save_bbox(record["action_uid"], record["elements"], img_width, img_height, current_dir, class_names)

        with open("coco8.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    finally:
        driver.close()

    if "--zip" in sys.argv:
        shutil.make_archive("CV_WebIdentification", "zip", ROOT_DIR)

    if "--clean" in sys.argv:
        shutil.rmtree(ROOT_DIR)
