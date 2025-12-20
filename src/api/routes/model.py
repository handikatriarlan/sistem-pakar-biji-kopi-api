"""
Model management endpoints.
"""

from fastapi import APIRouter, status, BackgroundTasks
from datetime import datetime
import pandas as pd
import os

from src.schemas.response import ModelInfoResponse, TrainingResponse
from src.models.backpropagation import neural_network
from src.core.config import settings, BACKPROPAGATION_CONFIG, FEATURE_COLUMNS
from src.core.exceptions import (
    DatasetNotFoundException, TrainingException, ModelNotTrainedException
)

router = APIRouter(prefix="/model", tags=["Model Management"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model Information",
    description="Get detailed information about the trained neural network model"
)
async def get_model_info():
    """
    Get comprehensive information about the Backpropagation Neural Network model.
    
    Returns:
    - Model architecture details
    - Training metrics (accuracy, precision, recall, F1)
    - Network configuration
    - Training history
    """
    if not neural_network.is_trained:
        raise ModelNotTrainedException()
    
    model_info = neural_network.get_model_info()
    
    return ModelInfoResponse(
        success=True,
        message="Model information retrieved successfully",
        timestamp=datetime.now(),
        data=model_info
    )


@router.get(
    "/architecture",
    status_code=status.HTTP_200_OK,
    summary="Get Network Architecture",
    description="Get detailed neural network architecture information"
)
async def get_architecture():
    """
    Get detailed Backpropagation Neural Network architecture.
    
    Shows:
    - Layer configuration
    - Neuron counts
    - Activation functions
    - Total parameters
    """
    architecture = neural_network.network_architecture
    config = BACKPROPAGATION_CONFIG
    
    return {
        "success": True,
        "message": "Network architecture retrieved",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "algorithm": "Backpropagation Neural Network",
            "architecture": architecture,
            "training_config": {
                "learning_rate": config["learning_rate"],
                "max_iterations": config["max_iterations"],
                "optimizer": config["solver"],
                "activation": config["activation"],
                "regularization_alpha": config["alpha"],
                "early_stopping": config["early_stopping"]
            },
            "description": {
                "forward_propagation": "Input → Hidden Layers (ReLU) → Output (Softmax)",
                "backward_propagation": "Compute gradients via chain rule, update weights",
                "optimization": "Adam optimizer with adaptive learning rate"
            }
        }
    }


@router.post(
    "/train",
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Train Model",
    description="Train or retrain the Backpropagation Neural Network model"
)
async def train_model():
    """
    Train the Backpropagation Neural Network model.
    
    Training Process:
    1. Load and preprocess dataset
    2. Split into training/test sets (80/20)
    3. Train using backpropagation algorithm
    4. Evaluate on test set
    5. Save trained model
    
    Returns training metrics and model performance.
    """
    dataset_path = settings.DATASET_PATH
    
    if not os.path.exists(dataset_path):
        raise DatasetNotFoundException(dataset_path)
    
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Train model
        training_result = neural_network.train(df)
        
        # Save model
        neural_network.save()
        
        return TrainingResponse(
            success=True,
            message="Model berhasil ditraining menggunakan Backpropagation",
            timestamp=datetime.now(),
            data={
                "training_completed": True,
                "samples_used": neural_network.metadata.get("training_samples"),
                "metrics": training_result["metrics"],
                "training_history": training_result["training_history"],
                "network_architecture": training_result["network_architecture"],
                "grade_distribution": neural_network.metadata.get("grade_distribution")
            }
        )
        
    except Exception as e:
        raise TrainingException(f"Training failed: {str(e)}")


@router.post(
    "/retrain",
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrain Model",
    description="Retrain the model with fresh data"
)
async def retrain_model():
    """Alias for train endpoint - retrains the model."""
    return await train_model()


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Get Training Metrics",
    description="Get detailed training and evaluation metrics"
)
async def get_metrics():
    """
    Get detailed model performance metrics.
    
    Includes:
    - Accuracy, Precision, Recall, F1 Score
    - Confusion Matrix
    - Cross-validation scores
    - Per-class metrics
    """
    if not neural_network.is_trained:
        raise ModelNotTrainedException()
    
    return {
        "success": True,
        "message": "Metrics retrieved successfully",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "overall_metrics": {
                "accuracy": neural_network.metrics.get("accuracy"),
                "precision_macro": neural_network.metrics.get("precision_macro"),
                "recall_macro": neural_network.metrics.get("recall_macro"),
                "f1_macro": neural_network.metrics.get("f1_macro")
            },
            "cross_validation": {
                "cv_accuracy_mean": neural_network.metrics.get("cv_accuracy_mean"),
                "cv_accuracy_std": neural_network.metrics.get("cv_accuracy_std")
            },
            "confusion_matrix": neural_network.metrics.get("confusion_matrix"),
            "per_class_metrics": neural_network.metrics.get("classification_report"),
            "training_info": {
                "iterations": neural_network.training_history.get("n_iterations"),
                "final_loss": neural_network.training_history.get("final_loss")
            }
        }
    }


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get Model Status",
    description="Check current model status"
)
async def get_model_status():
    """Get current model training status."""
    return {
        "success": True,
        "message": "Model status retrieved",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "is_trained": neural_network.is_trained,
            "training_date": neural_network.metadata.get("training_date"),
            "training_samples": neural_network.metadata.get("training_samples"),
            "model_files_exist": {
                "model": os.path.exists(settings.MODEL_PATH),
                "scaler": os.path.exists(settings.SCALER_PATH),
                "metadata": os.path.exists(settings.METADATA_PATH)
            }
        }
    }
