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

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/workspaces/WebIdentification/containers/model_backend/models/yolo.pt"))
if not MODEL_PATH.is_file() or MODEL_PATH.suffix.lower() != ".pt":
    raise ValueError(f"MODEL_PATH must point to an existing .pt file, got {MODEL_PATH}")


def _build_model():
    return YOLO(MODEL_PATH) if MODEL_PATH.name.startswith("yolo") else RTDETR(MODEL_PATH)


def _predict_once(pil_image: Image.Image):
    model = _build_model()
    return model(pil_image)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

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
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
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
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))