FROM python:3.12

WORKDIR /app
COPY ./containers/model_backend/requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


COPY ./containers/model_backend/models/ /app/models/
ENV MODEL_DIR=/app/models/
COPY ./src/webidentification/backend/server.py .
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["fastapi", "run", "server.py", "--host", "0.0.0.0", "--port", "8000"]
