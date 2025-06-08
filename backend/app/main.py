from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
from PIL import Image
import io
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from .schemas import PredictionResult, ModelInfo
from .models import Base, PredictionRecord
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0
    return image_array

@app.post("/predict/", response_model=PredictionResult)
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        processed_image = preprocess_image(contents)
        
        # Load model from MLflow
        model_uri = "models:/mnist_cnn/Production"
        model = mlflow.tensorflow.load_model(model_uri)
        
        prediction = model.predict(processed_image)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        # Save prediction to database
        db = SessionLocal()
        db_record = PredictionRecord(
            predicted_digit=predicted_digit,
            confidence=confidence,
            model_version="1.0"
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        db.close()
        
        return {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "prediction_id": db_record.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predictions/", response_model=List[PredictionResult])
async def get_predictions(limit: int = 10):
    db = SessionLocal()
    records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit).all()
    db.close()
    return records

@app.get("/model-info/", response_model=ModelInfo)
async def get_model_info():
    client = mlflow.tracking.MlflowClient()
    model = client.get_registered_model("mnist_cnn")
    return {
        "name": model.name,
        "latest_version": model.latest_versions[0].version,
        "description": model.description
    }