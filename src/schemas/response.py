"""
Response schemas for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class GradeEnum(str, Enum):
    """Coffee quality grade enumeration."""
    A = "A"
    B = "B"
    C = "C"


class BaseResponse(BaseModel):
    """Base response schema with common fields."""
    success: bool = Field(..., description="Indicates if request was successful")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ErrorDetail(BaseModel):
    """Error detail schema."""
    error_code: str = Field(..., description="Unique error code")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default={}, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = Field(default=False)
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.now)


class FeatureAnalysis(BaseModel):
    """Analysis of individual feature."""
    name: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value")
    status: str = Field(..., description="Status: excellent, good, needs_improvement")
    recommendation: Optional[str] = Field(None, description="Improvement recommendation")


class NetworkLayerInfo(BaseModel):
    """Information about neural network layer."""
    layer_name: str
    neurons: int
    activation: str


class BackpropagationDetails(BaseModel):
    """Details about backpropagation computation."""
    input_neurons: int = Field(..., description="Number of input neurons")
    hidden_layers: List[NetworkLayerInfo] = Field(..., description="Hidden layer configuration")
    output_neurons: int = Field(..., description="Number of output neurons")
    activation_function: str = Field(..., description="Activation function used")
    output_activation: str = Field(..., description="Output layer activation")
    learning_rate: float = Field(..., description="Learning rate used")
    total_parameters: int = Field(..., description="Total trainable parameters")


class PredictionResult(BaseModel):
    """Single prediction result."""
    grade: GradeEnum = Field(..., description="Predicted quality grade (A, B, or C)")
    grade_label: str = Field(..., description="Human-readable grade label")
    grade_description: str = Field(..., description="Grade description")
    predicted_score: float = Field(..., description="Predicted total cup points (0-100)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each grade")


class PredictionResponse(BaseResponse):
    """Response schema for single prediction."""
    data: Optional[Dict[str, Any]] = Field(None, description="Prediction data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Prediksi berhasil",
                "timestamp": "2024-01-15T10:30:00",
                "data": {
                    "prediction": {
                        "grade": "A",
                        "grade_label": "Specialty Grade",
                        "grade_description": "Kualitas excellent",
                        "predicted_score": 87.5,
                        "confidence": 0.92,
                        "probabilities": {"A": 0.92, "B": 0.07, "C": 0.01}
                    },
                    "feature_analysis": [],
                    "recommendations": [],
                    "backpropagation_info": {}
                }
            }
        }


class BatchPredictionResponse(BaseResponse):
    """Response schema for batch prediction."""
    data: Optional[Dict[str, Any]] = Field(None, description="Batch prediction data")


class ModelInfoResponse(BaseResponse):
    """Response schema for model information."""
    data: Optional[Dict[str, Any]] = Field(None, description="Model information")


class TrainingResponse(BaseResponse):
    """Response schema for model training."""
    data: Optional[Dict[str, Any]] = Field(None, description="Training results")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_status: str = Field(..., description="Model status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)


class GradeInfoResponse(BaseResponse):
    """Response schema for grade information."""
    data: Optional[Dict[str, Any]] = Field(None, description="Grade information")


class FeatureInfoResponse(BaseResponse):
    """Response schema for feature information."""
    data: Optional[Dict[str, Any]] = Field(None, description="Feature information")
