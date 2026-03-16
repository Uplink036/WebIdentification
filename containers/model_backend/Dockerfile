FROM python:3.12

WORKDIR /app
COPY ./containers/model_backend/requirements.txt .
RUN pip install -r requirements.txt


COPY ./containers/model_backend/models/ ./models/
ENV MODEL_PATH=/app/models/yolo.pt
COPY ./src/webidentification/backend/server.py .
CMD ["fastapi", "run", "server.py", "--host", "0.0.0.0", "--port", "8000"]
