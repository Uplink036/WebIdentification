import base64
import csv
import io
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from math import ceil

from neo4j import GraphDatabase
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

URI = os.getenv("URI", "bolt://localhost:7687")
AUTH = (os.getenv("USERNAME", "neo4j"), os.getenv("PASSWORD", "password"))

MAX_WORKERS = 8
DEFAULT_MAX_WIDTH = 640
DEFAULT_TILE_HEIGHT = 640

DEFAULT_OUTPUT_DIR = pathlib.Path("./clustimage_images")
DEFAULT_BY_SPLIT = False


def safe_filename(action_uid: str, suffix: str = "") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in action_uid)
    return f"screenshot_{cleaned}{suffix}.png"


def resize_with_aspect_ratio(image: Image.Image, max_width: int) -> Image.Image:
    image_width, image_height = image.size
    if image_width == max_width:
        return image
    ratio = image_width / float(max_width)
    new_height = int(round(image_height / ratio, 0))
    return image.resize((max_width, new_height), Image.LANCZOS)


def split_vertical_tiles(image: Image.Image, tile_height: int) -> list[Image.Image]:
    image_width, image_height = image.size
    n_tiles = ceil(image_height / float(tile_height))
    tiles: list[Image.Image] = []
    for index in range(n_tiles):
        y0 = tile_height * index
        y1 = min(tile_height * (index + 1), image_height)
        tiles.append(image.crop((0, y0, image_width, y1)))
    return tiles


def export_action_images(
    action_uid: str,
    action_type: str,
    domain: str,
    website: str,
    screenshot_b64: str,
    output_dir: pathlib.Path,
    by_split: bool,
) -> list[dict[str, str | int]]:
    image_bytes = base64.b64decode(screenshot_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized = resize_with_aspect_ratio(image, DEFAULT_MAX_WIDTH)

    split_dir = output_dir / action_type if by_split else output_dir
    split_dir.mkdir(parents=True, exist_ok=True)

    tiles = split_vertical_tiles(resized, DEFAULT_TILE_HEIGHT)
    tile_records: list[dict[str, str | int]] = []
    max_workers = max(1, min(MAX_WORKERS, len(tiles)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        targets: list[pathlib.Path] = []
        for index, tile in enumerate(tiles):
            suffix = f"_{index}"
            filename = safe_filename(action_uid, suffix)
            target = split_dir / filename
            targets.append(target)
            futures.append(executor.submit(tile.save, target))

        for future in futures:
            future.result()

    for index, target in enumerate(targets):
        tile_records.append(
            {
                "image": str(target.relative_to(output_dir)),
                "type": action_type,
                "domain": domain,
                "website": website,
            }
        )
    return tile_records


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = output_dir / "metadata.csv"
    metadata_rows: list[dict[str, str | int]] = []

    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with driver.session() as session:
            records = list(session.run("""
                MATCH (t:Task)-[:HAS_ACTION]->(a:Action)
                WITH a,
                     collect(DISTINCT t.domain)[0] AS domain,
                     collect(DISTINCT t.website)[0] AS website
                RETURN a.id AS action_uid,
                       a.type AS type,
                       domain,
                       website
                ORDER BY action_uid
                """))
            total = len(records)

            for record in tqdm(
                records, total=total, desc="Exporting images", unit="img"
            ):
                action_uid = record["action_uid"]
                action_type = record["type"] or "unknown"
                domain = record["domain"] or "unknown"
                website = record["website"] or "unknown"
                screenshot_record = session.run(
                    "MATCH (a:Action {id: $action_uid}) RETURN a.screenshot_b64 AS screenshot",
                    action_uid=action_uid,
                ).single()
                screenshot = (
                    screenshot_record["screenshot"] if screenshot_record else None
                )

                if not action_uid or not screenshot:
                    continue

                try:
                    rows = export_action_images(
                        action_uid=action_uid,
                        action_type=action_type,
                        domain=domain,
                        website=website,
                        screenshot_b64=screenshot,
                        output_dir=output_dir,
                        by_split=DEFAULT_BY_SPLIT,
                    )
                    metadata_rows.extend(rows)
                except Exception as exc:
                    print(f"Skipping action {action_uid}: {exc}")

        with metadata_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image", "type", "domain", "website"],
            )
            writer.writeheader()
            writer.writerows(metadata_rows)

        print(f"Export completed: {len(metadata_rows)} images")
        print(f"Images directory: {output_dir}")
        print(f"Metadata file: {metadata_file}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
