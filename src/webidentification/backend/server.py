import os
from io import BytesIO
from pathlib import Path

import dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from ultralytics import RTDETR, YOLO
import logging
import time

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("webidentification")

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
    try:
        if model_path.name.lower().startswith("yolo"):
            model = YOLO(str(model_path))
        else:
            model = RTDETR(str(model_path))
    except Exception as e:
        logger.error("model_load_failed %s", model_path.name, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_path.name}: {e}")
    logger.info("model_loaded %s", model_path.name)
    return model


def _build_model(model_name: str | None = None):
    range_models = _get_all_models()
    names = [model.name for model in range_models]
    if model_name:
        if model_name not in names:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found. Available models: {names}")
        model_path = range_models[names.index(model_name)]
        logger.info("model_selected %s", model_path.name)
    else:
        model_path = range_models[0]
        logger.info("model_selected %s", model_path.name)
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
    start = time.perf_counter()
    client = getattr(request, "client", None)
    client_ip = client.host if client else request.headers.get("x-forwarded-for", "unknown").split(",")[0].strip()
    logger.info("req /predict from=%s model=%s", client_ip, model or "default")
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
        duration = time.perf_counter() - start
        logger.info("done /predict from=%s model=%s dur=%.3f preds=%d", client_ip, model or "default", duration, len(predictions))
        return {"predictions": predictions}
    except UnidentifiedImageError:
        logger.warning("invalid_image from=%s", client_ip)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    except HTTPException as e:
        duration = time.perf_counter() - start
        logger.info("err /predict from=%s model=%s dur=%.3f status=%s", client_ip, model or "default", duration, getattr(e, "status_code", ""))
        raise e
    except Exception as e:
        duration = time.perf_counter() - start
        logger.error("err_unexpected /predict from=%s model=%s dur=%.3f", client_ip, model or "default", duration, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_draw")
async def predict_draw(request: Request, model: str | None = None):
    start = time.perf_counter()
    client = getattr(request, "client", None)
    client_ip = client.host if client else request.headers.get("x-forwarded-for", "unknown").split(",")[0].strip()
    logger.info("req /predict_draw from=%s model=%s", client_ip, model or "default")
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
        duration = time.perf_counter() - start
        logger.info("done /predict_draw from=%s model=%s dur=%.3f", client_ip, model or "default", duration)
        return Response(content=output.getvalue(), media_type="image/png")
    except UnidentifiedImageError:
        logger.warning("invalid_image from=%s", client_ip)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    except HTTPException as e:
        duration = time.perf_counter() - start
        logger.info("err /predict_draw from=%s model=%s dur=%.3f status=%s", client_ip, model or "default", duration, getattr(e, "status_code", ""))
        raise e
    except Exception as e:
        duration = time.perf_counter() - start
        logger.error("err_unexpected /predict_draw from=%s model=%s dur=%.3f", client_ip, model or "default", duration, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    