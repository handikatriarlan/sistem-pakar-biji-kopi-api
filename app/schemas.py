"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, List


class CoffeeFeatures(BaseModel):
    """Input features for coffee quality prediction."""
    aroma: float = Field(..., ge=0, le=10, description="Aroma score (0-10)")
    flavor: float = Field(..., ge=0, le=10, description="Flavor score (0-10)")
    aftertaste: float = Field(..., ge=0, le=10, description="Aftertaste score (0-10)")
    acidity: float = Field(..., ge=0, le=10, description="Acidity score (0-10)")
    body: float = Field(..., ge=0, le=10, description="Body score (0-10)")
    balance: float = Field(..., ge=0, le=10, description="Balance score (0-10)")
    uniformity: float = Field(..., ge=0, le=10, description="Uniformity score (0-10)")
    clean_cup: float = Field(..., ge=0, le=10, description="Clean Cup score (0-10)")
    sweetness: float = Field(..., ge=0, le=10, description="Sweetness score (0-10)")
    moisture_percentage: float = Field(..., ge=0, le=20, description="Moisture percentage (0-20)")

    class Config:
        json_schema_extra = {
            "example": {
                "aroma": 8.0,
                "flavor": 8.0,
                "aftertaste": 7.5,
                "acidity": 8.0,
                "body": 7.5,
                "balance": 8.0,
                "uniformity": 10.0,
                "clean_cup": 10.0,
                "sweetness": 10.0,
                "moisture_percentage": 11.0
            }
        }


class PredictionResult(BaseModel):
    """Prediction result with grade and recommendations."""
    grade: str = Field(..., description="Quality grade (A, B, or C)")
    predicted_score: float = Field(..., description="Predicted total cup points")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    quality_label: str = Field(..., description="Quality label description")
    recommendations: List[str] = Field(..., description="Quality improvement recommendations")


class ModelInfo(BaseModel):
    """Model information and metrics."""
    model_type: str
    accuracy: float
    features_used: List[str]
    grade_distribution: dict
    training_samples: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
