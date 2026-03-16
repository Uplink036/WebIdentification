FROM python:3.12

WORKDIR /app

COPY ./containers/frontend/requirements.txt .
RUN pip install -r requirements.txt

COPY ./src/webidentification/frontend/ ./

CMD ["fastapi", "run", "server.py", "--host", "0.0.0.0", "--port", "8010"]
