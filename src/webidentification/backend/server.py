import os
from io import BytesIO
from pathlib import Path

import dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from ultralytics import RTDETR, YOLO

dotenv.load_dotenv()

MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        "/workspaces/WebIdentification/containers/model_backend/models/",
    )
)


def _get_all_models() -> list[Path]:
    return list(MODEL_DIR.glob("*.pt"))

assert len(_get_all_models()) > 0, f"No model files found in {MODEL_DIR}. Please ensure at least one .pt file is present."

def _load_model_from_path(model_path: Path):
    """Load the given model on request (no caching).

    A fresh model instance is created each time this is called.
    """
    if model_path.name.lower().startswith("yolo"):
        model = YOLO(str(model_path))
    else:
        model = RTDETR(str(model_path))
    print(f"Loaded model: {model_path}")
    return model


def _build_model(model_name: str | None = None):
    range_models = _get_all_models()
    names = [model.name for model in range_models]
    if model_name:
        if model_name not in names:
            raise ValueError(f"Model {model_name} not found in {MODEL_DIR}. Available models: {names}")
        model_path = range_models[names.index(model_name)]
        print(f"Using model: {model_path}")
    else:
        model_path = range_models[0]
        print(f"No model specified. Defaulting to: {model_path}")
    return _load_model_from_path(model_path)

def _predict_once(pil_image: Image.Image, model_name: str | None = None):
    model = _build_model(model_name)
    return model(pil_image)


app = FastAPI()

@app.get("/healthcheck")
async def health():
    return {"message": "Ok"}

@app.get("/get_models")
async def get_models():
    return {"models": [model.name for model in _get_all_models()]}

@app.post("/predict")
async def predict(request: Request, model: str | None = None):
    try:
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Request body is empty")

        pil_image = Image.open(BytesIO(data)).convert("RGB")
        results = await run_in_threadpool(_predict_once, pil_image, model)
        boxes = results[0].boxes
        cls_list = boxes.cls.tolist()
        conf_list = boxes.conf.tolist()
        xywhn_list = boxes.xywhn.tolist()

        predictions = [
            {"class": int(c), "confidence": float(conf), "box": {"xywhn": box}}
            for c, conf, box in zip(cls_list, conf_list, xywhn_list)
        ]
        return {"predictions": predictions}
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_draw")
async def predict_draw(request: Request, model: str | None = None):
    try:
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Request body is empty")

        pil_image = Image.open(BytesIO(data)).convert("RGB")
        results = await run_in_threadpool(_predict_once, pil_image, model)

        im_bgr = results[0].plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        output = BytesIO()
        im_rgb.save(output, format="PNG")
        return Response(content=output.getvalue(), media_type="image/png")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
