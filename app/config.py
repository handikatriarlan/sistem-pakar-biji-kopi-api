"""Configuration settings for the Coffee Quality Expert System."""

# Feature columns used for prediction
FEATURE_COLUMNS = [
    "Aroma", "Flavor", "Aftertaste", "Acidity", 
    "Body", "Balance", "Uniformity", "Clean Cup", 
    "Sweetness", "Moisture Percentage"
]

# Grade classification thresholds based on Total Cup Points
GRADE_THRESHOLDS = {
    "A": 85.0,  # Specialty Grade (>=85)
    "B": 80.0,  # Premium Grade (80-84.99)
    "C": 0.0    # Below Premium (<80)
}

# Neural Network configuration
NN_CONFIG = {
    "hidden_layers": (64, 32, 16),
    "max_iter": 1000,
    "learning_rate_init": 0.001,
    "activation": "relu",
    "solver": "adam",
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.1
}

# Model paths
MODEL_PATH = "models/coffee_quality_model.joblib"
SCALER_PATH = "models/scaler.joblib"
