#!/usr/bin/env python3
"""Tool to retrieve screenshots from Mind2Web database and save to COCO format."""
import random
import base64
import io
import pathlib
import shutil
import sys
import os
import yaml
import json
from neo4j import GraphDatabase
from PIL import Image, ImageFile
from tqdm import tqdm
from math import floor, ceil

URI = os.getenv("URI", "bolt://localhost:7687")
AUTH = (os.getenv("USERNAME", "neo4j"), os.getenv("PASSWORD", "password"))


SPLITS = ["train", "test_domain", "test_task", "test_website"]

ROOT_DIR = pathlib.Path("./CV_WebIdentification")
TRAIN_DIR = ROOT_DIR / "train"
TEST_DIR = ROOT_DIR / "test"
VAL_DIR = ROOT_DIR / "val"

MAX_WIDTH = 1920
MAX_HEIGHT = 1080

ELEMENT_FILTER = {
    "button": "button",
    "a": "button",
}

def get_safe_filename(action_uid, format="png"):
    filename = f"screenshot_{action_uid}.{format}"
    return filename

def save_screenshot(action_uid, screenshot_b64, dir):
    img_data = base64.b64decode(screenshot_b64)
    img = Image.open(io.BytesIO(img_data))
    resized_image = resize_with_aspect_ratio(img)
    for index, image in enumerate(unstitch_image(resized_image)):
        filename = get_safe_filename(f"{action_uid}_{index}", "png")
        image.save(dir / filename)
    return img.size

def unstitch_image(image: ImageFile) -> list[ImageFile]:                                                                                                                    
    max_width = MAX_WIDTH                                                                                                                                           
    max_height = MAX_HEIGHT                                                                                                                                         
                                                                                                                                                                    
    image_width, image_height = image.size[0], image.size[1]                                                                                                        
    print(image_width, image_height)                                                                                                                                
    new_images = ceil(image_height / max_height)                                                                                                                    
    images = [                                                                                                                                                      
        image.crop((0, max_height * i, max_width, max_height * (i + 1)))                                                                                            
        for i in range(0, new_images)                                                                                                                               
    ]                                                                                                                                                               
    return images                                                                                                                                                   

def save_bbox(action_uid, elements, img_width, img_height, dir, class_names):
    new_width, new_height = get_resized_width_and_height(img_width, img_height)
    bbox_bins = [[] for _ in range(0, ceil(new_height/MAX_HEIGHT))]
    for elem in elements:
        if elem["tag"] not in ELEMENT_FILTER:
            continue
        attrs = json.loads(elem["attributes"])
        bbox_str = attrs.get("bounding_box_rect")
        if not bbox_str:
            continue
        x_center, y_center, width, height = map(float, bbox_str.split(","))
        x_center = x_center * (new_width / img_width)
        y_center = y_center * (new_height / img_height)
        width = width * (new_width / img_width)
        height = height * (new_height / img_height)
        if x_center < 0 or x_center > new_width or y_center < 0 or y_center > new_height:
            continue
        
        class_name = ELEMENT_FILTER[elem["tag"]]
        class_id = class_names.index(class_name)
        bin_number = floor(y_center/MAX_HEIGHT)

        if bin_number < 0 or bin_number >= len(bbox_bins):
            continue

        slice_x_center = round(x_center / MAX_WIDTH, 5)
        slice_y_center = round((y_center - (bin_number * MAX_HEIGHT)) / MAX_HEIGHT, 5)
        slice_width_norm = round(width / MAX_WIDTH, 5)
        slice_height_norm = round(height / MAX_HEIGHT, 5)

        bbox_bins[bin_number].append(
            (class_id, slice_x_center, slice_y_center, slice_width_norm, slice_height_norm)
        )

    for index, bbox_bin in enumerate(bbox_bins):
        if not bbox_bin:
            os.remove(dir / get_safe_filename(f"{action_uid}_{index}", "png"))
            continue
        with open(dir / get_safe_filename(f"{action_uid}_{index}", "txt"), "w") as f:
            for class_id, slice_x_center, slice_y_center, slice_width_norm, slice_height_norm in bbox_bin:
                f.write(f"{class_id} {slice_x_center} {slice_y_center} {slice_width_norm} {slice_height_norm}\n")

def get_resized_width_and_height(image_width: int, image_height:int) -> tuple[int, int]:
    width_ratio = image_width / MAX_WIDTH
    new_height = image_height / width_ratio
    return MAX_WIDTH, int(round(new_height,0))

def resize_with_aspect_ratio(image: ImageFile) -> ImageFile:
    image_width, image_height = image.size[0], image.size[1]
    new_width, new_height = get_resized_width_and_height(image_width, image_height)
    return image.resize((new_width, new_height), Image.LANCZOS)

def get_current_dir(split):
    if split == "test_domain":
        return VAL_DIR
    elif split in ["test_task", "test_website"]:
        return TEST_DIR
    elif split == "train":
        return TRAIN_DIR
    else:
        raise ValueError(f"Unknown split: {split}")

if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=AUTH)
    data = {"path": "CV_WebIdentification", "train": "train", "test": "test", "val": "val"}
    
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with driver.session() as session:
            unique_class_names = sorted(set(ELEMENT_FILTER.values()))
            data["names"] = {i: name for i, name in enumerate(unique_class_names)}
            class_names = unique_class_names

        with driver.session() as session:
            result_ids = session.run("MATCH (a:Action) RETURN a.id AS action_uid")
            for record in tqdm(result_ids):
                action_uid = record["action_uid"]
                result_action = session.run("""
                    MATCH (a:Action {id: $action_uid})
                    OPTIONAL MATCH (a)-[:TARGETS]->(target:Element)
                    OPTIONAL MATCH (a)-[:HAS_CANDIDATE]->(candidate:Element)
                    RETURN a.id AS action_uid, a.screenshot_b64 AS screenshot, 
                           a.type AS split, collect(DISTINCT target) + collect(DISTINCT candidate) AS elements
                """, action_uid=action_uid)
                record = result_action.single()
                split = record["split"]
                current_dir = get_current_dir(split)
                
                img_width, img_height = save_screenshot(record["action_uid"], record["screenshot"], current_dir)
                save_bbox(record["action_uid"], record["elements"], img_width, img_height, current_dir, class_names)

        with open("cv_webidentification.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    finally:
        driver.close()

    if "--zip" in sys.argv:
        shutil.make_archive("CV_WebIdentification", "zip", ROOT_DIR)

    if "--clean" in sys.argv:
        shutil.rmtree(ROOT_DIR)
