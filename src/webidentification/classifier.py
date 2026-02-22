from ultralytics import YOLO


class WebElementClassifier:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path or "yolo11n-cls.pt")

    def classify(self, image_path):
        results = self.model(image_path)
        return results
