FROM python:3.9-slim

WORKDIR /app

RUN pip install mlflow==1.20.2 psycopg2-binary==2.9.1 boto3==1.18.42

CMD ["mlflow", "server", \
     "--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}", \
     "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}", \
     "--host", "0.0.0.0", \
     "--port", "5000"]