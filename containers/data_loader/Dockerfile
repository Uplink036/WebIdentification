FROM python:3.14

ENV URI=bolt://localhost:7687
ENV USERNAME=neo4j
ENV PASSWORD=password

WORKDIR /app 
COPY ./containers/data_loader/requirements.txt .
RUN pip install -r requirements.txt

COPY src/webidentification/pipeline/ingest_neo4j.py .
CMD ["python", "ingest_neo4j.py"]
