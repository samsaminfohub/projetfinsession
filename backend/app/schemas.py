from pydantic import BaseModel
from datetime import datetime

class PredictionResult(BaseModel):
    id: int
    predicted_digit: int
    confidence: float
    model_version: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class ModelInfo(BaseModel):
    name: str
    latest_version: str
    description: str