FROM tensorflow/tensorflow:2.6.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# This container will exit after training completes
CMD ["python", "train.py"]