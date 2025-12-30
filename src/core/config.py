"""
Configuration settings for the Coffee Quality Expert System.
"""

from pydantic_settings import BaseSettings
from typing import Tuple
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    APP_NAME: str = "Coffee Quality Expert System API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Sistem Pakar Penilaian Kualitas Biji Kopi menggunakan Backpropagation Neural Network"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    
    # Dataset Path
    DATASET_PATH: str = "data/df_arabica_clean.csv"
    
    # Model Paths
    MODEL_DIR: str = "trained_models"
    MODEL_PATH: str = "trained_models/backpropagation_model.joblib"
    SCALER_PATH: str = "trained_models/scaler.joblib"
    METADATA_PATH: str = "trained_models/metadata.joblib"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Feature columns used for prediction
FEATURE_COLUMNS = [
    "Aroma", "Flavor", "Aftertaste", "Acidity",
    "Body", "Balance", "Uniformity", "Clean Cup",
    "Sweetness", "Moisture Percentage"
]

# Grade classification thresholds based on Total Cup Points (SCA Standard)
GRADE_THRESHOLDS = {
    "A": {"min": 85.0, "max": 100.0, "label": "Specialty Grade", "description": "Kualitas excellent, memenuhi standar specialty coffee internasional"},
    "B": {"min": 80.0, "max": 84.99, "label": "Premium Grade", "description": "Kualitas baik, di atas standar komersial"},
    "C": {"min": 0.0, "max": 79.99, "label": "Below Premium Grade", "description": "Kualitas standar, perlu peningkatan signifikan"}
}

# Backpropagation Neural Network Configuration
BACKPROPAGATION_CONFIG = {
    # Network Architecture
    "input_layer_size": len(FEATURE_COLUMNS),  # 10 input neurons
    "hidden_layers": (64, 32, 16),  # 3 hidden layers
    "output_layer_size": 3,  # 3 classes (A, B, C)
    
    # Learning Parameters
    "learning_rate": 0.001,
    "max_iterations": 1000,
    "tolerance": 1e-4,
    
    # Activation Function
    "activation": "relu",  # ReLU for hidden layers
    "output_activation": "softmax",  # Softmax for output layer
    
    # Optimization
    "solver": "adam",  # Adam optimizer (adaptive learning rate)
    "batch_size": "auto",
    
    # Regularization
    "alpha": 0.0001,  # L2 regularization
    "early_stopping": False,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    
    # Random State for reproducibility
    "random_state": 42
}


settings = Settings()
