"""A tool to show the boundingboxes of a image (and predictions?)"""

import pillow
import argparse
from ultralytics import YOLO

def get_image(image_path) -> pillow.Image:
    """Load an image from the specified path."""
    try:
        image = pillow.Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def get_bounding_boxes(predictions_path) -> list:
    """Load the data next to the image."""
    try:
        with open(predictions_path, 'r') as file:
            bounding_boxes = []
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    bbox = {
                        'class': parts[0],
                        'x_min': int(parts[1]),
                        'y_min': int(parts[2]),
                        'x_max': int(parts[3]),
                        'y_max': int(parts[4])
                    }
                    bounding_boxes.append(bbox)
            return bounding_boxes
    except Exception as e:
        print(f"Error loading bounding boxes: {e}")
        return []

def get_predictions(model_path, image_path):
    """Get predictions from the model."""
    try:
        model = YOLO(model_path)
        results = model(image_path)
        return results
    except Exception as e:
        print(f"Error loading model or getting predictions: {e}")
        return None        

def draw_on_image(image, bounding_boxes, colour="red") -> pillow.Image:
    """Draw bounding boxes on the image."""
    draw = pillow.ImageDraw.Draw(image)
    for bbox in bounding_boxes:
        draw.rectangle([(bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max'])], outline=colour, width=2)
        draw.text((bbox['x_min'], bbox['y_min'] - 10), bbox['class'], fill=colour)
    return image


def main():
    parser = argparse.ArgumentParser(description="Display an image with bounding boxes.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("--predictions", type=str, help="Path to a model file (optional).")
    args = parser.parse_args()

    if args.image_path is None:
        print("Please provide a path to an image file.")
        return
    image = get_image(args.image_path)
    if image is None:
        return

    bbox_info_path = args.image_path.suffix(".txt")
    if bbox_info_path.exists():
        bounding_boxes = get_bounding_boxes(bbox_info_path)
        image = draw_on_image(image, bounding_boxes, colour="blue")


    if args.predictions is not None and args.predictions.exists():
        predictions = get_predictions(args.predictions, args.image_path)
        if predictions is not None:
            image = draw_on_image(image, predictions, colour="red")

    image.show()

if __name__ == "__main__":
    main()