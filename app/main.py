"""
Coffee Quality Expert System API

Sistem Pakar Penilaian Kualitas Biji Kopi menggunakan 
Metode Backpropagation Neural Network.

Author: Coffee Quality Expert System
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import pandas as pd
import os

from app.schemas import CoffeeFeatures, PredictionResult, ModelInfo, HealthResponse
from app.model import coffee_model
from app.config import FEATURE_COLUMNS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    # Try to load existing model
    if not coffee_model.load_model():
        # Train new model if not exists
        print("Training new model...")
        train_model_from_csv()
    print("Model ready!")
    yield
    # Cleanup on shutdown
    print("Shutting down...")


app = FastAPI(
    title="Coffee Quality Expert System API",
    description="""
    ## Sistem Pakar Penilaian Kualitas Biji Kopi
    
    API ini menggunakan **Backpropagation Neural Network** untuk memprediksi 
    kualitas biji kopi berdasarkan karakteristik sensorik.
    
    ### Fitur:
    - **Prediksi Grade**: Klasifikasi kualitas (A, B, C)
    - **Rekomendasi**: Saran peningkatan kualitas
    - **Confidence Score**: Tingkat kepercayaan prediksi
    
    ### Grade Classification:
    - **Grade A** (≥85 points): Specialty Grade - Excellent Quality
    - **Grade B** (80-84.99 points): Premium Grade - Good Quality
    - **Grade C** (<80 points): Standard Grade - Below Premium
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def train_model_from_csv():
    """Train model from CSV dataset."""
    csv_path = "df_arabica_clean.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    coffee_model.train(df)
    coffee_model.save_model()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Coffee Quality Expert System API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "/api/v1/predict",
            "model_info": "/api/v1/model/info",
            "health": "/api/v1/health",
            "retrain": "/api/v1/model/retrain"
        }
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and model health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=coffee_model.is_trained,
        version="1.0.0"
    )


@app.post("/api/v1/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_quality(features: CoffeeFeatures):
    """
    Predict coffee quality grade based on sensory features.
    
    Input features:
    - **aroma**: Aroma score (0-10)
    - **flavor**: Flavor score (0-10)
    - **aftertaste**: Aftertaste score (0-10)
    - **acidity**: Acidity score (0-10)
    - **body**: Body score (0-10)
    - **balance**: Balance score (0-10)
    - **uniformity**: Uniformity score (0-10)
    - **clean_cup**: Clean Cup score (0-10)
    - **sweetness**: Sweetness score (0-10)
    - **moisture_percentage**: Moisture percentage (0-20)
    
    Returns predicted grade, score, confidence, and recommendations.
    """
    if not coffee_model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum siap. Silakan coba lagi nanti."
        )
    
    try:
        features_dict = {
            "aroma": features.aroma,
            "flavor": features.flavor,
            "aftertaste": features.aftertaste,
            "acidity": features.acidity,
            "body": features.body,
            "balance": features.balance,
            "uniformity": features.uniformity,
            "clean_cup": features.clean_cup,
            "sweetness": features.sweetness,
            "moisture_percentage": features.moisture_percentage
        }
        
        result = coffee_model.predict(features_dict)
        return PredictionResult(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/api/v1/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the trained model."""
    if not coffee_model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum ditraining."
        )
    
    return ModelInfo(
        model_type="Backpropagation Neural Network (MLPClassifier)",
        accuracy=coffee_model.metrics.get("accuracy", 0),
        features_used=FEATURE_COLUMNS,
        grade_distribution=coffee_model.grade_distribution,
        training_samples=coffee_model.training_samples
    )


@app.post("/api/v1/model/retrain", tags=["Model"])
async def retrain_model():
    """Retrain the model with the dataset."""
    try:
        train_model_from_csv()
        return {
            "message": "Model berhasil ditraining ulang",
            "metrics": coffee_model.metrics,
            "grade_distribution": coffee_model.grade_distribution
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training error: {str(e)}"
        )


@app.get("/api/v1/grades", tags=["Reference"])
async def get_grade_info():
    """Get grade classification information."""
    return {
        "grades": [
            {
                "grade": "A",
                "min_score": 85.0,
                "label": "Specialty Grade",
                "description": "Kualitas excellent, memenuhi standar specialty coffee"
            },
            {
                "grade": "B", 
                "min_score": 80.0,
                "max_score": 84.99,
                "label": "Premium Grade",
                "description": "Kualitas baik, di atas standar komersial"
            },
            {
                "grade": "C",
                "max_score": 79.99,
                "label": "Standard Grade",
                "description": "Kualitas standar, perlu peningkatan"
            }
        ]
    }


@app.get("/api/v1/features", tags=["Reference"])
async def get_feature_info():
    """Get information about input features."""
    return {
        "features": [
            {"name": "aroma", "description": "Intensitas dan kualitas aroma", "range": "0-10"},
            {"name": "flavor", "description": "Rasa keseluruhan", "range": "0-10"},
            {"name": "aftertaste", "description": "Rasa yang tertinggal setelah minum", "range": "0-10"},
            {"name": "acidity", "description": "Tingkat keasaman", "range": "0-10"},
            {"name": "body", "description": "Ketebalan dan tekstur", "range": "0-10"},
            {"name": "balance", "description": "Keseimbangan rasa", "range": "0-10"},
            {"name": "uniformity", "description": "Konsistensi antar cup", "range": "0-10"},
            {"name": "clean_cup", "description": "Kebersihan rasa", "range": "0-10"},
            {"name": "sweetness", "description": "Tingkat kemanisan", "range": "0-10"},
            {"name": "moisture_percentage", "description": "Persentase kelembaban biji", "range": "0-20"}
        ]
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )
