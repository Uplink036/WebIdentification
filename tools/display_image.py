"""A tool to show the boundingboxes of a image (and predictions?)"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


@dataclass
class BoundingBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

def get_image(image_path: Path) -> Image.Image | None:
    """Load an image from the specified path."""
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def get_bounding_boxes(predictions_path: Path) -> list[BoundingBox]:
    """Load the data next to the image."""
    try:
        with open(predictions_path, "r", encoding="utf-8") as file:
            bounding_boxes = []
            for line in file:
                parts = line.strip().split()
                bbox = BoundingBox(
                    class_id=int(parts[0]),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                )
                bounding_boxes.append(bbox)
            return bounding_boxes
    except Exception as e:
        print(f"Error loading bounding boxes: {e}")
        return []


def get_predictions(image_path: Path, model_path: Path) -> list[BoundingBox] | None:
    """Get predictions from the model."""
    try:
        model = YOLO(model_path)
        results = model(image_path)
        detected_boxes = []
        if results is not None and len(results) > 0:
            boxes_xywhn = results[0].boxes.xywhn.tolist()
            classes = results[0].boxes.cls.tolist()

            for box_xywhn, cls_id in zip(boxes_xywhn, classes):
                detected_boxes.append(
                    BoundingBox(
                        class_id=int(cls_id),
                        x_center=float(box_xywhn[0]),
                        y_center=float(box_xywhn[1]),
                        width=float(box_xywhn[2]),
                        height=float(box_xywhn[3]),
                    )
                )
        return detected_boxes
    except Exception as e:
        print(f"Error loading model or getting predictions: {e}")
        return None


def draw_on_image(
    image: Image.Image, bounding_boxes: list[BoundingBox], colour: str = "red"
) -> Image.Image:
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    for bbox in bounding_boxes:
        x_min = (bbox.x_center - bbox.width / 2) * img_w
        y_min = (bbox.y_center - bbox.height / 2) * img_h
        x_max = (bbox.x_center + bbox.width / 2) * img_w
        y_max = (bbox.y_center + bbox.height / 2) * img_h
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=colour, width=2)
        draw.text((x_min, max(0, y_min)), str(bbox.class_id), fill=colour)
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Display an image with bounding boxes."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        help="Path to a model file (optional).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save rendered output.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)

    if not image_path.exists():
        print(f"Image file does not exist: {image_path}")
        return

    image = get_image(image_path)
    if image is None:
        return

    bbox_info_path = image_path.with_suffix(".txt")
    if bbox_info_path.exists():
        bounding_boxes = get_bounding_boxes(bbox_info_path)
        image = draw_on_image(image, bounding_boxes, colour="blue")

    if args.model is not None:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
        else:
            detected_boxes = get_predictions(image_path, model_path)
            image = draw_on_image(image, detected_boxes, colour="red")

    output_path = (
        Path(args.output)
        if args.output
        else image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")
    )

    image.save(output_path)
    print(f"Saved rendered image to: {output_path}")


if __name__ == "__main__":
    main()
