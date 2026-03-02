import os
import wandb
from dotenv import load_dotenv
from ultralytics import settings, YOLO

# View all settings
load_dotenv()
settings.update({"wandb": True})
model = YOLO("yolo26n.pt")

# WANDB login
wandb.login(key=os.environ.get("WANDB_API_KEY"))


# Train and Fine-Tune the Model
model.train(data="coco8.yaml", epochs=100, project="ultralytics", name="yolo26n")
