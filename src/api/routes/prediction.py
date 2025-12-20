"""
Prediction endpoints for coffee quality assessment.
"""

from fastapi import APIRouter, status, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from datetime import datetime

from src.schemas.request import CoffeeQualityInput, BatchPredictionInput
from src.schemas.response import (
    PredictionResponse, BatchPredictionResponse, ErrorResponse
)
from src.models.backpropagation import neural_network
from src.core.exceptions import ModelNotTrainedException, PredictionException

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Coffee Quality",
    description="Predict coffee quality grade using Backpropagation Neural Network",
    responses={
        200: {
            "description": "Prediction successful",
            "model": PredictionResponse
        },
        422: {
            "description": "Validation error - invalid input values"
        },
        503: {
            "description": "Model not ready"
        }
    }
)
async def predict_quality(input_data: CoffeeQualityInput):
    """
    Predict coffee quality grade based on sensory evaluation features.
    
    The prediction uses a Backpropagation Neural Network trained on
    CQI (Coffee Quality Institute) arabica coffee dataset.
    
    **Input Features (SCA Cupping Protocol):**
    - aroma: Intensitas dan kualitas aroma (0-10)
    - flavor: Rasa keseluruhan (0-10)
    - aftertaste: Rasa yang tertinggal (0-10)
    - acidity: Tingkat keasaman (0-10)
    - body: Ketebalan dan tekstur (0-10)
    - balance: Keseimbangan rasa (0-10)
    - uniformity: Konsistensi antar cup (0-10)
    - clean_cup: Kebersihan rasa (0-10)
    - sweetness: Tingkat kemanisan (0-10)
    - moisture_percentage: Kelembaban biji (0-20%)
    
    **Output:**
    - grade: A (Specialty), B (Premium), or C (Below Premium)
    - confidence: Prediction confidence (0-1)
    - recommendations: Quality improvement suggestions
    """
    if not neural_network.is_trained:
        raise ModelNotTrainedException()
    
    try:
        # Convert input to dictionary
        features = input_data.to_dict()
        
        # Perform prediction using backpropagation network
        result = neural_network.predict(features)
        
        return PredictionResponse(
            success=True,
            message="Prediksi berhasil dilakukan",
            timestamp=datetime.now(),
            data=result
        )
        
    except ValueError as e:
        raise ModelNotTrainedException(str(e))
    except Exception as e:
        raise PredictionException(f"Error during prediction: {str(e)}")


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Prediction",
    description="Predict quality for multiple coffee samples at once",
    responses={
        200: {"description": "Batch prediction successful"},
        422: {"description": "Validation error"},
        503: {"description": "Model not ready"}
    }
)
async def predict_batch(input_data: BatchPredictionInput):
    """
    Predict quality for multiple coffee samples.
    
    Maximum 100 samples per request.
    """
    if not neural_network.is_trained:
        raise ModelNotTrainedException()
    
    try:
        samples = [sample.to_dict() for sample in input_data.samples]
        results = neural_network.predict_batch(samples)
        
        # Summary statistics
        grades = [r["prediction"]["grade"] for r in results]
        summary = {
            "total_samples": len(results),
            "grade_distribution": {
                "A": grades.count("A"),
                "B": grades.count("B"),
                "C": grades.count("C")
            },
            "average_confidence": sum(r["prediction"]["confidence"] for r in results) / len(results)
        }
        
        return BatchPredictionResponse(
            success=True,
            message=f"Batch prediction berhasil untuk {len(results)} sampel",
            timestamp=datetime.now(),
            data={
                "predictions": results,
                "summary": summary
            }
        )
        
    except Exception as e:
        raise PredictionException(f"Batch prediction error: {str(e)}")


@router.post(
    "/analyze",
    status_code=status.HTTP_200_OK,
    summary="Detailed Analysis",
    description="Get detailed analysis of coffee quality features"
)
async def analyze_features(input_data: CoffeeQualityInput):
    """
    Perform detailed analysis of coffee quality features.
    
    Returns comprehensive breakdown of each feature with
    status indicators and specific recommendations.
    """
    if not neural_network.is_trained:
        raise ModelNotTrainedException()
    
    try:
        features = input_data.to_dict()
        result = neural_network.predict(features)
        
        # Calculate overall score breakdown
        sensory_scores = {
            "aroma": features["aroma"],
            "flavor": features["flavor"],
            "aftertaste": features["aftertaste"],
            "acidity": features["acidity"],
            "body": features["body"],
            "balance": features["balance"],
            "uniformity": features["uniformity"],
            "clean_cup": features["clean_cup"],
            "sweetness": features["sweetness"]
        }
        
        total_sensory = sum(sensory_scores.values())
        
        return {
            "success": True,
            "message": "Analisis berhasil",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "prediction": result["prediction"],
                "score_breakdown": {
                    "sensory_total": round(total_sensory, 2),
                    "individual_scores": sensory_scores,
                    "moisture": features["moisture_percentage"]
                },
                "feature_analysis": result["feature_analysis"],
                "recommendations": result["recommendations"],
                "quality_summary": {
                    "strengths": [
                        f["name"] for f in result["feature_analysis"] 
                        if f["status"] == "excellent"
                    ],
                    "areas_for_improvement": [
                        f["name"] for f in result["feature_analysis"]
                        if f["status"] == "needs_improvement"
                    ]
                }
            }
        }
        
    except Exception as e:
        raise PredictionException(f"Analysis error: {str(e)}")
