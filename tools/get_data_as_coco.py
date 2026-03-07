#!/usr/bin/env python3
"""Tool to retrieve screenshots from Mind2Web database and save to COCO format."""

import base64
import io
import json
import os
import pathlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from math import ceil, floor
import yaml
from neo4j import GraphDatabase
from PIL import Image, ImageFile
from tqdm import tqdm

URI = os.getenv("URI", "bolt://localhost:7687")
AUTH = (os.getenv("USERNAME", "neo4j"), os.getenv("PASSWORD", "password"))

SPLITS = ["train", "test_domain", "test_task", "test_website"]

ROOT_DIR = pathlib.Path("./CV_WebIdentification")
TRAIN_DIR = ROOT_DIR / "train"
TEST_DIR = ROOT_DIR / "test"
VAL_DIR = ROOT_DIR / "val"

MAX_WORKERS = 8
MAX_WIDTH = 1920
MAX_HEIGHT = 1080

ELEMENT_FILTER = {
    "button": "button",
    "a": "button",
}


def get_safe_filename(action_uid: str, format: str = "png") -> str:
    filename = f"screenshot_{action_uid}.{format}"
    return filename


def get_current_dir(split: str) -> pathlib.Path:
    if split == "test_domain":
        return VAL_DIR
    elif split in ["test_task", "test_website"]:
        return TEST_DIR
    elif split == "train":
        return TRAIN_DIR
    else:
        raise ValueError(f"Unknown split: {split}")


def get_resized_width_and_height(
    image_width: int, image_height: int
) -> tuple[int, int]:
    width_ratio = image_width / MAX_WIDTH
    new_height = image_height / width_ratio
    return MAX_WIDTH, int(round(new_height, 0))


def resize_with_aspect_ratio(image: ImageFile) -> ImageFile:
    image_width, image_height = image.size[0], image.size[1]
    new_width, new_height = get_resized_width_and_height(image_width, image_height)
    return image.resize((new_width, new_height), Image.LANCZOS)


def unstitch_image(image: ImageFile) -> list[ImageFile]:
    _, image_height = image.size
    new_images = ceil(image_height / MAX_HEIGHT)

    images = [
        image.crop((0, MAX_HEIGHT * i, MAX_WIDTH, MAX_HEIGHT * (i + 1)))
        for i in range(0, new_images)
    ]

    return images


def save_screenshot(action_uid: str, screenshot_b64: str, dir: pathlib.Path) -> tuple[int, int] :
    img_data = base64.b64decode(screenshot_b64)
    img = Image.open(io.BytesIO(img_data))
    resized_image = resize_with_aspect_ratio(img)
    unstitch_images = unstitch_image(resized_image)

    for index, image in enumerate(unstitch_images):
        filename = get_safe_filename(f"{action_uid}_{index}", "png")
        image.save(dir / filename)

    return img.size


def is_bin_number_out_of_bounds(bbox_bins: list[list], bin_number: int) -> bool:
    return bin_number < 0 or bin_number >= len(bbox_bins)


def determine_y_bin_from_center(y_center: float) -> int:
    return floor(y_center / MAX_HEIGHT)


def get_class_id_from_element(class_names: list[str], elem: dict) -> int:
    class_name = ELEMENT_FILTER[elem["tag"]]
    class_id = class_names.index(class_name)
    return class_id


def is_within_image_bounds(new_width: int, new_height: int, x_center: float, y_center: float) -> bool:
    return (
        x_center >= 0
        and x_center <= new_width
        and y_center >= 0
        and y_center <= new_height
    )


def resize_bounding_box(img_width: int, img_height: int, new_width: int, new_height: int, bbox_str: str) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = map(float, bbox_str.split(","))
    x_center = x_center * (new_width / img_width)
    y_center = y_center * (new_height / img_height)
    width = width * (new_width / img_width)
    height = height * (new_height / img_height)
    return x_center, y_center, width, height


def normalize_bounding_box(x_center, y_center, width, height, bin_number):
    slice_x_center = round(x_center / MAX_WIDTH, 5)
    slice_y_center = round((y_center - (bin_number * MAX_HEIGHT)) / MAX_HEIGHT, 5)
    slice_width_norm = round(width / MAX_WIDTH, 5)
    slice_height_norm = round(height / MAX_HEIGHT, 5)
    return slice_x_center, slice_y_center, slice_width_norm, slice_height_norm


def save_bbox(action_uid: str, elements: list[dict], img_width: int, img_height: int, dir: pathlib.Path, class_names: list[str]):
    new_width, new_height = get_resized_width_and_height(img_width, img_height)
    bbox_bins = [[] for _ in range(0, ceil(new_height / MAX_HEIGHT))]
    for elem in elements:
        if elem["tag"] not in ELEMENT_FILTER:
            continue
        attrs = json.loads(elem["attributes"])
        bbox_str = attrs.get("bounding_box_rect")
        if not bbox_str:
            continue

        x_center, y_center, width, height = resize_bounding_box(
            img_width, img_height, new_width, new_height, bbox_str
        )
        if not is_within_image_bounds(new_width, new_height, x_center, y_center):
            continue
        bin_number = determine_y_bin_from_center(y_center)
        if is_bin_number_out_of_bounds(bbox_bins, bin_number):
            continue
        class_id = get_class_id_from_element(class_names, elem)

        slice_x_center, slice_y_center, slice_width_norm, slice_height_norm = (
            normalize_bounding_box(x_center, y_center, width, height, bin_number)
        )
        bbox_bins[bin_number].append(
            (
                class_id,
                slice_x_center,
                slice_y_center,
                slice_width_norm,
                slice_height_norm,
            )
        )

    for index, bbox_bin in enumerate(bbox_bins):
        if not bbox_bin:
            os.remove(dir / get_safe_filename(f"{action_uid}_{index}", "png"))
            continue
        with open(dir / get_safe_filename(f"{action_uid}_{index}", "txt"), "w") as f:
            for (
                class_id,
                slice_x_center,
                slice_y_center,
                slice_width_norm,
                slice_height_norm,
            ) in bbox_bin:
                f.write(
                    f"{class_id} {slice_x_center} {slice_y_center} {slice_width_norm} {slice_height_norm}\n"
                )


def process_action(record: dict, class_names: list[str]):
    split = record["split"]
    current_dir = get_current_dir(split)

    img_width, img_height = save_screenshot(
        record["action_uid"],
        record["screenshot"],
        current_dir,
    )
    save_bbox(
        record["action_uid"],
        record["elements"],
        img_width,
        img_height,
        current_dir,
        class_names,
    )


def iter_action_records(session) -> dict:
    result_ids = session.run("MATCH (a:Action) RETURN a.id AS action_uid LIMIT 200")
    for record in result_ids:
        action_uid = record["action_uid"]
        result_action = session.run(
            """
            MATCH (a:Action {id: $action_uid})
            OPTIONAL MATCH (a)-[:TARGETS]->(target:Element)
            OPTIONAL MATCH (a)-[:HAS_CANDIDATE]->(candidate:Element)
            RETURN a.id AS action_uid, a.screenshot_b64 AS screenshot,
                   a.type AS split, collect(DISTINCT target) + collect(DISTINCT candidate) AS elements
        """,
            action_uid=action_uid,
        )
        action_record = result_action.single()

        elements = [
            {
                "tag": elem.get("tag"),
                "attributes": elem.get("attributes"),
            }
            for elem in action_record["elements"]
            if elem is not None
        ]

        yield {
            "action_uid": action_record["action_uid"],
            "screenshot": action_record["screenshot"],
            "split": action_record["split"],
            "elements": elements,
        }


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    data = {
        "path": "CV_WebIdentification",
        "train": "train",
        "test": "test",
        "val": "val",
    }

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with driver.session() as session:
            unique_class_names = sorted(set(ELEMENT_FILTER.values()))
            data["names"] = {i: name for i, name in enumerate(unique_class_names)}
            class_names = unique_class_names

        with driver.session() as session:
            action_records = iter_action_records(session)
            process_action_with_classes = partial(process_action, class_names=class_names)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for _ in tqdm(
                    executor.map(process_action_with_classes, action_records),
                    desc="Processing actions",
                ):
                    pass

        with open("cv_webidentification.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    finally:
        driver.close()

    if "--zip" in sys.argv:
        shutil.make_archive("CV_WebIdentification", "zip", ROOT_DIR)

    if "--clean" in sys.argv:
        shutil.rmtree(ROOT_DIR)


if __name__ == "__main__":
    main()
