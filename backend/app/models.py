from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func
from .database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    predicted_digit = Column(Integer)
    confidence = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TrainingLog(Base):
    __tablename__ = "model_training_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    version = Column(String)
    accuracy = Column(Float)
    training_time = Column(DateTime(timezone=True), server_default=func.now())