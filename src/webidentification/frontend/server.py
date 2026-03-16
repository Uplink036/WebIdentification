import base64
import os
import time
from io import BytesIO
from pathlib import Path

import dotenv
import httpx
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options

dotenv.load_dotenv()

DEFAULT_MODEL_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8000/predict_draw")
DEFAULT_SELENIUM_REMOTE_URL = os.getenv("SELENIUM_REMOTE_URL", "")
UI_PATH = Path(__file__).with_name("ui.html")

def _create_driver(width: int = 1920, height: int = 1080) -> webdriver.Remote:
	options = Options()
	options.add_argument("--headless=new")
	options.add_argument("--disable-gpu")
	options.add_argument("--disable-dev-shm-usage")
	options.add_argument("--no-sandbox")
	options.add_argument("--hide-scrollbars")
	options.add_argument(f"--window-size={width},{height}")

	selenium_remote_url = DEFAULT_SELENIUM_REMOTE_URL
	if not selenium_remote_url:
		raise ValueError("SELENIUM_REMOTE_URL must be set for remote WebDriver")

	return webdriver.Remote(command_executor=selenium_remote_url, options=options)


def _capture_page_slices(driver: webdriver.Remote, scroll_delay: float = 0.3) -> list[tuple[int, bytes]]:
	device_pixel_ratio = int(driver.execute_script("return window.devicePixelRatio") or 1)

	total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
	viewport_height = driver.execute_script("return window.innerHeight")
	if not isinstance(total_height, (int, float)) or not isinstance(viewport_height, (int, float)):
		raise ValueError("Unable to determine page dimensions for screenshot capture")

	total_height = int(total_height)
	viewport_height = int(viewport_height)
	if total_height <= 0 or viewport_height <= 0:
		raise ValueError("Page dimensions are invalid for screenshot capture")

	offset = 0
	slices: list[tuple[int, bytes]] = []
	while offset < total_height:
		if offset + viewport_height > total_height:
			offset = total_height - viewport_height

		driver.execute_script("window.scrollTo({0}, {1})".format(0, offset))
		time.sleep(scroll_delay)

		png_bytes = driver.get_screenshot_as_png()
		if device_pixel_ratio > 1:
			# Normalize slice pixels when the browser renders at a higher DPR.
			img = Image.open(BytesIO(png_bytes)).convert("RGB")
			img = img.resize((img.width // device_pixel_ratio, img.height // device_pixel_ratio))
			buf = BytesIO()
			img.save(buf, format="PNG")
			png_bytes = buf.getvalue()

		slices.append((offset, png_bytes))
		offset += viewport_height
		if offset >= total_height:
			break

	return slices

def _stitch_slices(slices: list[tuple[int, bytes]]) -> bytes:
	if not slices:
		raise ValueError("No screenshot slices were captured")

	opened_images: list[tuple[int, Image.Image]] = []
	for offset, png_bytes in slices:
		opened_images.append((offset, Image.open(BytesIO(png_bytes)).convert("RGB")))

	max_width = max(img.width for _, img in opened_images)
	total_height = max(offset + img.height for offset, img in opened_images)
	stitched = Image.new("RGB", (max_width, total_height))

	for offset, img in sorted(opened_images, key=lambda x: x[0]):
		stitched.paste(img, (0, offset))

	buf = BytesIO()
	stitched.save(buf, format="PNG")
	return buf.getvalue()


def _take_screenshot(url: str) -> list[tuple[int, bytes]]:
	driver = None
	try:
		driver = _create_driver()
		driver.get(url)
		slices = _capture_page_slices(driver)
		return slices
	except WebDriverException as exc:
		message = getattr(exc, "msg", str(exc))
		raise ValueError(f"Selenium failed to render the page: {message}") from exc
	finally:
		if driver is not None:
			driver.quit()


async def _forward_to_model(model_url: str, image_bytes: bytes, timeout_seconds: float) -> bytes:
	async with httpx.AsyncClient(timeout=timeout_seconds) as client:
		result = await client.post(
			model_url,
			content=image_bytes,
			headers={"Content-Type": "application/octet-stream"},
		)

	if result.status_code >= 400:
		detail = result.text or "Model server request failed"
		raise HTTPException(status_code=result.status_code, detail=detail)

	return result.content


app = FastAPI(title="WebIdentification Frontend")


@app.get("/")
async def root() -> FileResponse:
	return FileResponse(UI_PATH, media_type="text/html")


@app.get("/health")
async def health() -> dict[str, str]:
	return {"message": "Frontend service is running"}


@app.post("/screenshot")
async def screenshot(url: str = Query(..., description="URL of the page to capture")) -> Response:
	try:
		slices = await run_in_threadpool(_take_screenshot, url)
		stitched = await run_in_threadpool(_stitch_slices, slices)
		return Response(content=stitched, media_type="image/png")
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/screenshot/forward")
async def screenshot_and_forward(url: str = Query(..., description="URL of the page to capture and forward")) -> Response:
	try:
		slices = await run_in_threadpool(_take_screenshot, url)
		model_slices: list[tuple[int, bytes]] = []
		for offset, png_bytes in slices:
			model_png_bytes = await _forward_to_model(
				model_url=DEFAULT_MODEL_URL,
				image_bytes=png_bytes,
				timeout_seconds=30,
			)
			model_slices.append((offset, model_png_bytes))

		model_stitch = await run_in_threadpool(_stitch_slices, model_slices)
		return Response(content=model_stitch, media_type="image/png")
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except httpx.HTTPError as exc:
		raise HTTPException(status_code=502, detail=f"Failed to reach model server: {exc}") from exc