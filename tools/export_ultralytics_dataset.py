import base64
import io
import json
import os
import pathlib
import shutil
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor

import yaml
from neo4j import GraphDatabase
from PIL import Image
from tqdm import tqdm

URI = os.getenv("URI", "bolt://localhost:7687")
AUTH = (os.getenv("USERNAME", "neo4j"), os.getenv("PASSWORD", "password"))

SPLITS = ["train", "test_domain", "test_task", "test_website"]

ROOT_DIR = pathlib.Path("./CV_WebIdentification")
TRAIN_DIR = ROOT_DIR / "train"
TEST_DIR = ROOT_DIR / "test"
VAL_DIR = ROOT_DIR / "val"

MAX_WIDTH = 1920
MAX_HEIGHT = 1080

MAX_WORKERS = 8

ELEMENT_FILTER = {
    "button": "button",
    "a": "button",
}

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    RUNNING = False
    print("Interrupt received, stopping...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_safe_filename(action_uid: str, format: str = "png") -> str:
    filename = f"screenshot_{action_uid}.{format}"
    return filename


def unstitch_image(image: Image.Image) -> list[Image.Image]:
    max_width = MAX_WIDTH
    max_height = MAX_HEIGHT

    image_height = image.size[1]
    new_images = ceil(image_height / max_height)
    images = [
        image.crop((0, max_height * i, max_width, max_height * (i + 1)))
        for i in range(0, new_images)
    ]
    return images


def save_screenshot(
    action_uid: str, screenshot_b64: str, dir: pathlib.Path
) -> tuple[int, int]:
    img_data = base64.b64decode(screenshot_b64)
    img = Image.open(io.BytesIO(img_data))
    resized_image = resize_with_aspect_ratio(img)

    image_slices = list(enumerate(unstitch_image(resized_image)))
    if not image_slices:
        print(f"Error: No image slices generated for action {action_uid}")
        return img.size
    max_workers = min(MAX_WORKERS, len(image_slices))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, image in image_slices:
            filename = get_safe_filename(f"{action_uid}_{index}", "png")
            futures.append(executor.submit(image.save, dir / filename))

        for future in futures:
            future.result()
    return img.size


def save_bbox(
    action_uid: str,
    elements: list[dict],
    img_width: int,
    img_height: int,
    dir: pathlib.Path,
    class_names: list[str],
) -> None:
    new_width, new_height = get_resized_width_and_height(img_width, img_height)
    bbox_bins = [[] for _ in range(0, ceil(new_height / MAX_HEIGHT))]
    for elem in elements:
        if elem["tag"] not in ELEMENT_FILTER:
            continue
        attrs = json.loads(elem["attributes"])
        bbox_str = attrs.get("bounding_box_rect")
        if not bbox_str:
            continue

        x_min, y_min, width, height = resize_bounding_box(
            img_width, img_height, new_width, new_height, bbox_str
        )
        x_center, y_center, width, height = convert_tlwh_to_xywh(
            x_min, y_min, width, height
        )
        if not is_within_image_bounds(new_width, new_height, x_center, y_center):
            continue
        bin_number = determine_y_bin_from_center(y_center)
        if is_bin_number_out_of_bounds(bbox_bins, bin_number):
            continue
        class_id = get_class_id_from_element(class_names, elem["tag"])

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


def is_bin_number_out_of_bounds(bbox_bins: list, bin_number: int) -> bool:
    return bin_number < 0 or bin_number >= len(bbox_bins)


def determine_y_bin_from_center(y_center: float) -> int:
    return floor(y_center / MAX_HEIGHT)


def get_class_id_from_element(class_names: list[str], elem_tag: str) -> int:
    class_name = ELEMENT_FILTER[elem_tag]
    class_id = class_names.index(class_name)
    return class_id


def is_within_image_bounds(
    new_width: int, new_height: int, x_center: float, y_center: float
) -> bool:
    return (
        x_center >= 0
        and x_center <= new_width
        and y_center >= 0
        and y_center <= new_height
    )


def convert_tlwh_to_xywh(
    x_min: float, y_min: float, width: float, height: float
) -> tuple[float, float, float, float]:
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return x_center, y_center, width, height


def resize_bounding_box(
    img_width: int, img_height: int, new_width: int, new_height: int, bbox_str: str
) -> tuple[float, float, float, float]:
    x_min, y_min, width, height = map(float, bbox_str.split(","))
    x_min = x_min * (new_width / img_width)
    y_min = y_min * (new_height / img_height)
    width = width * (new_width / img_width)
    height = height * (new_height / img_height)
    return x_min, y_min, width, height


def normalize_bounding_box(
    x_center: float, y_center: float, width: float, height: float, bin_number: int
) -> tuple[float, float, float, float]:
    slice_x_center = round(x_center / MAX_WIDTH, 5)
    slice_y_center = round((y_center - (bin_number * MAX_HEIGHT)) / MAX_HEIGHT, 5)
    slice_width_norm = round(width / MAX_WIDTH, 5)
    slice_height_norm = round(height / MAX_HEIGHT, 5)
    return slice_x_center, slice_y_center, slice_width_norm, slice_height_norm


def get_resized_width_and_height(
    image_width: int, image_height: int
) -> tuple[int, int]:
    width_ratio = image_width / MAX_WIDTH
    new_height = image_height / width_ratio
    return MAX_WIDTH, int(round(new_height, 0))


def resize_with_aspect_ratio(image: Image.Image) -> Image.Image:
    image_width, image_height = image.size[0], image.size[1]
    new_width, new_height = get_resized_width_and_height(image_width, image_height)
    return image.resize((new_width, new_height), Image.LANCZOS)


def get_current_dir(split: str) -> pathlib.Path:
    if split == "test_website":
        return VAL_DIR
    elif split == "test_domain":
        return TEST_DIR
    elif split == "train":
        return TRAIN_DIR
    elif split == "test_task":
        return None
    else:
        raise ValueError(f"Unknown split: {split}")


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
            result_ids = session.run("MATCH (a:Action) RETURN a.id AS action_uid")
            for record in tqdm(result_ids, desc="Processing actions", unit=" db-image"):
                if not RUNNING:
                    print("Stopping early due to interrupt.")
                    break
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
                record = result_action.single()
                split = record["split"]
                current_dir = get_current_dir(split)
                if current_dir is None:
                    continue

                img_width, img_height = save_screenshot(
                    record["action_uid"], record["screenshot"], current_dir
                )
                save_bbox(
                    record["action_uid"],
                    record["elements"],
                    img_width,
                    img_height,
                    current_dir,
                    class_names,
                )

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
