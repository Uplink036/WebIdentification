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

def _build_model(model_name: str = None):
    range_models = _get_all_models()
    names = [model.name for model in range_models]
    if model_name and model_name in names:
        model_index = names.index(model_name)
        model_path = _get_all_models()[model_index]
        print(f"Using model: {model_path}")
    else:
        raise ValueError(f"Model {model_name} not found in {MODEL_DIR}. Available models: {names}")

    if model_name is None:
        model_path = range_models[0]
        print(f"No model specified. Defaulting to: {model_path}")
    return (
        YOLO(MODEL_DIR) if MODEL_DIR.name.startswith("yolo") else RTDETR(MODEL_DIR)
    )

def _predict_once(pil_image: Image.Image):
    model = _build_model()
    return model(pil_image)

app = FastAPI()

@app.get("/healthcheck")
async def health():
    return {"message": "Ok"}

@app.get("/get_models")
async def get_models():
    return {"models": [model.name for model in _get_all_models()]}

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Request body is empty")

        pil_image = Image.open(BytesIO(data)).convert("RGB")
        results = await run_in_threadpool(_predict_once, pil_image)
        predictions = [
            {
                "class": int(cls),
                "confidence": float(conf),
                "box": {"xywhn": box},
            }
            for cls, conf, box in zip(
                results[0].boxes.cls,
                results[0].boxes.conf,
                results[0].boxes.xywhn.tolist(),
            )
        ]
        return {"predictions": predictions}
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_draw")
async def predict_draw(request: Request):
    try:
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Request body is empty")

        pil_image = Image.open(BytesIO(data)).convert("RGB")
        results = await run_in_threadpool(_predict_once, pil_image)

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
