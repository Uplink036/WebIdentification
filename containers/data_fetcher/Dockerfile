FROM python:3.14

ENV URI=bolt://localhost:7687
ENV USERNAME=neo4j
ENV PASSWORD=password

WORKDIR /app 
COPY ./containers/data_fetcher/requirements.txt .
RUN pip install -r requirements.txt

COPY ./tools/get_data_as_coco.py .
CMD ["python", "get_data_as_coco.py", "--zip", "--clean"]
