"""
Coffee Quality Expert System API

Main application entry point.
Sistem Pakar Penilaian Kualitas Biji Kopi menggunakan
Metode Backpropagation Neural Network.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import pandas as pd
import os
from datetime import datetime

from src.core.config import settings
from src.core.exceptions import BaseAPIException
from src.models.backpropagation import neural_network
from src.api.routes import health, prediction, model, reference


def train_model_on_startup():
    """Train model from dataset on startup if not already trained."""
    dataset_path = settings.DATASET_PATH
    
    if not os.path.exists(dataset_path):
        print(f"⚠ Dataset not found at {dataset_path}")
        return False
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"📊 Training model with {len(df)} samples...")
        neural_network.train(df)
        neural_network.save()
        print(f"✅ Model trained successfully! Accuracy: {neural_network.metrics.get('accuracy', 0):.2%}")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 Starting Coffee Quality Expert System API...")
    
    # Try to load existing model
    if neural_network.load():
        print("✅ Loaded existing trained model")
    else:
        print("📦 No existing model found, training new model...")
        train_model_on_startup()
    
    print(f"🌐 API ready at http://{settings.HOST}:{settings.PORT}")
    print(f"📚 Documentation at http://{settings.HOST}:{settings.PORT}/docs")
    
    yield
    
    print("👋 Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=f"""
## {settings.APP_DESCRIPTION}

### 🎯 Fitur Utama
- **Prediksi Grade Kualitas**: Klasifikasi kopi ke Grade A, B, atau C
- **Analisis Fitur**: Breakdown detail setiap atribut sensorik
- **Rekomendasi**: Saran peningkatan kualitas berbasis AI
- **Batch Prediction**: Prediksi multiple sampel sekaligus

### 🧠 Algoritma
Menggunakan **Backpropagation Neural Network** dengan arsitektur:
- Input Layer: 10 neurons (fitur sensorik)
- Hidden Layers: 64 → 32 → 16 neurons (ReLU activation)
- Output Layer: 3 neurons (Softmax untuk klasifikasi)
- Optimizer: Adam dengan adaptive learning rate

### 📊 Grade Classification (SCA Standard)
| Grade | Score Range | Label |
|-------|-------------|-------|
| **A** | ≥ 85 | Specialty Grade |
| **B** | 80 - 84.99 | Premium Grade |
| **C** | < 80 | Below Premium |

### 🔗 Quick Links
- [Predict Quality](/api/v1/predict) - POST
- [Model Info](/api/v1/model/info) - GET
- [Grade Reference](/api/v1/reference/grades) - GET
""",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "error_code": exc.error_code,
                "message": exc.detail["message"],
                "details": exc.details
            },
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " → ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "error_code": "VALIDATION_ERROR",
                "message": "Input validation failed",
                "details": {"errors": errors}
            },
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc)} if settings.DEBUG else {}
            },
            "timestamp": datetime.now().isoformat()
        }
    )


# Include Routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(prediction.router, prefix="/api/v1")
app.include_router(model.router, prefix="/api/v1")
app.include_router(reference.router, prefix="/api/v1")


# Root Endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information and available endpoints.
    """
    return {
        "success": True,
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "timestamp": datetime.now().isoformat(),
        "data": {
            "description": settings.APP_DESCRIPTION,
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json"
            },
            "endpoints": {
                "health": {
                    "check": "GET /api/v1/health",
                    "ready": "GET /api/v1/health/ready",
                    "live": "GET /api/v1/health/live"
                },
                "prediction": {
                    "predict": "POST /api/v1/predict",
                    "batch": "POST /api/v1/predict/batch",
                    "analyze": "POST /api/v1/predict/analyze"
                },
                "model": {
                    "info": "GET /api/v1/model/info",
                    "architecture": "GET /api/v1/model/architecture",
                    "train": "POST /api/v1/model/train",
                    "metrics": "GET /api/v1/model/metrics",
                    "status": "GET /api/v1/model/status"
                },
                "reference": {
                    "grades": "GET /api/v1/reference/grades",
                    "features": "GET /api/v1/reference/features",
                    "cupping_guide": "GET /api/v1/reference/cupping-guide"
                }
            }
        }
    }


@app.get("/api/v1", tags=["Root"])
async def api_root():
    """API v1 root endpoint."""
    return {
        "success": True,
        "message": "Coffee Quality Expert System API v1",
        "version": "1.0.0",
        "model_ready": neural_network.is_trained
    }
