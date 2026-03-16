import os
import time
from io import BytesIO
from pathlib import Path

import dotenv
import httpx
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, HttpUrl
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options

dotenv.load_dotenv()

DEFAULT_MODEL_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8000/predict_draw")
DEFAULT_SELENIUM_REMOTE_URL = os.getenv("SELENIUM_REMOTE_URL", "")
def _create_driver(width: int = 1080, height: int = 1920) -> webdriver.Remote:
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
@app.get("/health")
async def health() -> dict[str, str]:
	return {"message": "Frontend service is running"}
