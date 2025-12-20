"""
Backpropagation Neural Network Implementation for Coffee Quality Classification.

This module implements a Multi-Layer Perceptron (MLP) using the Backpropagation
algorithm for training. The network classifies coffee quality into grades A, B, or C.

Backpropagation Algorithm Steps:
1. Forward Pass: Input propagates through network, computing activations
2. Compute Error: Calculate loss at output layer
3. Backward Pass: Propagate error backwards, computing gradients
4. Update Weights: Adjust weights using gradient descent

Mathematical Foundation:
- Forward: z = Wx + b, a = activation(z)
- Error: δ_output = (y_pred - y_true) * activation'(z)
- Backprop: δ_hidden = (W^T · δ_next) * activation'(z)
- Update: W = W - learning_rate * δ · a^T
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime

from src.core.config import (
    FEATURE_COLUMNS, GRADE_THRESHOLDS, BACKPROPAGATION_CONFIG,
    settings
)


class BackpropagationNeuralNetwork:
    """
    Coffee Quality Classification using Backpropagation Neural Network.
    
    This class wraps scikit-learn's MLPClassifier which implements
    the backpropagation algorithm for training multi-layer perceptrons.
    
    Architecture:
    - Input Layer: 10 neurons (one per feature)
    - Hidden Layer 1: 64 neurons (ReLU activation)
    - Hidden Layer 2: 32 neurons (ReLU activation)  
    - Hidden Layer 3: 16 neurons (ReLU activation)
    - Output Layer: 3 neurons (Softmax activation for classification)
    
    Training Algorithm: Backpropagation with Adam optimizer
    """
    
    def __init__(self):
        """Initialize the neural network components."""
        self.model: Optional[MLPClassifier] = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model state
        self.is_trained = False
        self.training_history: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Network architecture info
        self.network_architecture = self._build_architecture_info()
    
    def _build_architecture_info(self) -> Dict[str, Any]:
        """Build network architecture information."""
        config = BACKPROPAGATION_CONFIG
        hidden_layers = config["hidden_layers"]
        
        # Calculate total parameters
        input_size = config["input_layer_size"]
        layers = [input_size] + list(hidden_layers) + [config["output_layer_size"]]
        
        total_params = 0
        for i in range(len(layers) - 1):
            # Weights + biases
            total_params += layers[i] * layers[i+1] + layers[i+1]
        
        return {
            "input_neurons": input_size,
            "hidden_layers": [
                {"layer": i+1, "neurons": n, "activation": config["activation"]}
                for i, n in enumerate(hidden_layers)
            ],
            "output_neurons": config["output_layer_size"],
            "activation_function": config["activation"],
            "output_activation": config["output_activation"],
            "learning_rate": config["learning_rate"],
            "optimizer": config["solver"],
            "total_parameters": total_params
        }
    
    def _assign_grade(self, total_cup_points: float) -> str:
        """
        Assign quality grade based on total cup points.
        
        SCA (Specialty Coffee Association) Standard:
        - Grade A (Specialty): >= 85 points
        - Grade B (Premium): 80-84.99 points
        - Grade C (Below Premium): < 80 points
        """
        if total_cup_points >= GRADE_THRESHOLDS["A"]["min"]:
            return "A"
        elif total_cup_points >= GRADE_THRESHOLDS["B"]["min"]:
            return "B"
        return "C"
    
    def _get_grade_info(self, grade: str) -> Dict[str, str]:
        """Get grade label and description."""
        info = GRADE_THRESHOLDS.get(grade, GRADE_THRESHOLDS["C"])
        return {
            "label": info["label"],
            "description": info["description"]
        }

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data for neural network training.
        
        Steps:
        1. Select relevant features
        2. Handle missing values with median imputation
        3. Create grade labels based on Total Cup Points
        4. Normalize features using StandardScaler (z-score normalization)
        
        Args:
            df: DataFrame with coffee quality data
            
        Returns:
            Tuple of (X_scaled, y_encoded, y_scores)
        """
        # Select features
        feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        X = df[feature_cols].copy()
        
        # Handle missing values with median imputation
        X = X.fillna(X.median())
        
        # Create grade labels from Total Cup Points
        y_grades = df["Total Cup Points"].apply(self._assign_grade)
        y_scores = df["Total Cup Points"].values
        
        # Store grade distribution
        self.metadata["grade_distribution"] = y_grades.value_counts().to_dict()
        self.metadata["feature_statistics"] = {
            col: {
                "mean": float(X[col].mean()),
                "std": float(X[col].std()),
                "min": float(X[col].min()),
                "max": float(X[col].max())
            }
            for col in feature_cols
        }
        
        # Normalize features using StandardScaler
        # z = (x - mean) / std
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels (A=0, B=1, C=2)
        y_encoded = self.label_encoder.fit_transform(y_grades)
        
        return X_scaled, y_encoded, y_scores
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the Backpropagation Neural Network.
        
        The training process uses the backpropagation algorithm:
        
        1. FORWARD PROPAGATION:
           - Input layer receives normalized features
           - Each hidden layer computes: z = W·x + b, a = ReLU(z)
           - Output layer computes: z = W·a + b, y = Softmax(z)
        
        2. COMPUTE LOSS:
           - Cross-entropy loss: L = -Σ y_true * log(y_pred)
        
        3. BACKWARD PROPAGATION:
           - Compute gradients: ∂L/∂W, ∂L/∂b for each layer
           - Output layer: δ = y_pred - y_true
           - Hidden layers: δ = (W_next^T · δ_next) * ReLU'(z)
        
        4. UPDATE WEIGHTS (Adam optimizer):
           - W = W - lr * m_hat / (sqrt(v_hat) + ε)
           - Where m, v are momentum estimates
        
        Args:
            df: Training DataFrame
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        config = BACKPROPAGATION_CONFIG
        
        # Preprocess data
        X, y, y_scores = self.preprocess_data(df)
        self.metadata["training_samples"] = len(X)
        self.metadata["training_date"] = datetime.now().isoformat()
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=config["random_state"],
            stratify=y
        )
        
        # Initialize MLPClassifier with backpropagation
        self.model = MLPClassifier(
            hidden_layer_sizes=config["hidden_layers"],
            activation=config["activation"],
            solver=config["solver"],
            alpha=config["alpha"],
            batch_size=config["batch_size"],
            learning_rate_init=config["learning_rate"],
            max_iter=config["max_iterations"],
            tol=config["tolerance"],
            early_stopping=config["early_stopping"],
            validation_fraction=config["validation_fraction"],
            n_iter_no_change=config["n_iter_no_change"],
            random_state=config["random_state"],
            verbose=False
        )
        
        # Train model using backpropagation
        self.model.fit(X_train, y_train)
        
        # Store training history
        self.training_history = {
            "loss_curve": self.model.loss_curve_ if hasattr(self.model, 'loss_curve_') else [],
            "n_iterations": self.model.n_iter_,
            "final_loss": float(self.model.loss_),
            "best_loss": float(min(self.model.loss_curve_)) if self.model.loss_curve_ else float(self.model.loss_)
        }
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        self.metrics["cv_accuracy_mean"] = float(cv_scores.mean())
        self.metrics["cv_accuracy_std"] = float(cv_scores.std())
        
        self.is_trained = True
        
        return {
            "training_history": self.training_history,
            "metrics": self.metrics,
            "network_architecture": self.network_architecture
        }

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict coffee quality grade using trained neural network.
        
        Forward Propagation Process:
        1. Normalize input features using fitted scaler
        2. Pass through hidden layers with ReLU activation
        3. Output layer produces class probabilities via Softmax
        4. Select class with highest probability
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Prediction result with grade, confidence, and analysis
        """
        if not self.is_trained:
            raise ValueError("Model belum ditraining")
        
        # Prepare input features in correct order
        feature_values = [
            features.get("aroma", 0),
            features.get("flavor", 0),
            features.get("aftertaste", 0),
            features.get("acidity", 0),
            features.get("body", 0),
            features.get("balance", 0),
            features.get("uniformity", 0),
            features.get("clean_cup", 0),
            features.get("sweetness", 0),
            features.get("moisture_percentage", 0)
        ]
        
        # Reshape for single prediction
        X = np.array(feature_values).reshape(1, -1)
        
        # Normalize using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Forward propagation through network
        grade_encoded = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Decode grade
        grade = self.label_encoder.inverse_transform([grade_encoded])[0]
        grade_info = self._get_grade_info(grade)
        
        # Get confidence (probability of predicted class)
        confidence = float(max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            cls: float(prob) 
            for cls, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        # Calculate estimated total cup points
        sensory_sum = sum(feature_values[:9])  # Sum of 9 sensory attributes
        predicted_score = min(100, max(0, sensory_sum))
        
        # Analyze features
        feature_analysis = self._analyze_features(features)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(features, grade)
        
        return {
            "prediction": {
                "grade": grade,
                "grade_label": grade_info["label"],
                "grade_description": grade_info["description"],
                "predicted_score": round(predicted_score, 2),
                "confidence": round(confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
            },
            "feature_analysis": feature_analysis,
            "recommendations": recommendations,
            "backpropagation_info": {
                "input_normalized": X_scaled[0].tolist(),
                "network_output": probabilities.tolist(),
                "architecture": self.network_architecture
            }
        }
    
    def predict_batch(self, samples: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Predict multiple samples at once."""
        return [self.predict(sample) for sample in samples]
    
    def _analyze_features(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Analyze individual features and provide status."""
        analysis = []
        
        thresholds = {
            "aroma": {"excellent": 8.0, "good": 7.0},
            "flavor": {"excellent": 8.0, "good": 7.0},
            "aftertaste": {"excellent": 7.5, "good": 7.0},
            "acidity": {"excellent": 8.0, "good": 7.0},
            "body": {"excellent": 8.0, "good": 7.0},
            "balance": {"excellent": 8.0, "good": 7.0},
            "uniformity": {"excellent": 10.0, "good": 9.0},
            "clean_cup": {"excellent": 10.0, "good": 9.0},
            "sweetness": {"excellent": 10.0, "good": 9.0},
            "moisture_percentage": {"excellent": 11.0, "good": 10.0}  # Ideal range
        }
        
        for feature, value in features.items():
            if feature not in thresholds:
                continue
                
            thresh = thresholds[feature]
            
            if feature == "moisture_percentage":
                # Special handling for moisture (ideal: 9-12%)
                if 9 <= value <= 12:
                    status = "excellent"
                elif 8 <= value <= 12.5:
                    status = "good"
                else:
                    status = "needs_improvement"
            else:
                if value >= thresh["excellent"]:
                    status = "excellent"
                elif value >= thresh["good"]:
                    status = "good"
                else:
                    status = "needs_improvement"
            
            analysis.append({
                "name": feature,
                "value": value,
                "status": status,
                "threshold_excellent": thresh["excellent"],
                "threshold_good": thresh["good"]
            })
        
        return analysis
    
    def _generate_recommendations(self, features: Dict[str, float], grade: str) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Feature-specific recommendations
        rec_map = {
            "aroma": (7.5, "Tingkatkan proses pengeringan dan penyimpanan untuk aroma yang lebih baik"),
            "flavor": (7.5, "Perhatikan profil roasting untuk meningkatkan flavor"),
            "aftertaste": (7.5, "Evaluasi kualitas biji mentah dan proses fermentasi untuk aftertaste lebih baik"),
            "acidity": (7.5, "Sesuaikan ketinggian tanam atau varietas untuk acidity optimal"),
            "body": (7.5, "Pertimbangkan metode processing berbeda (natural/honey) untuk body lebih kuat"),
            "balance": (7.5, "Fokus pada keseimbangan profil rasa secara keseluruhan"),
            "uniformity": (10.0, "Tingkatkan konsistensi dalam proses sorting dan grading"),
            "clean_cup": (10.0, "Perbaiki proses pencucian dan fermentasi untuk clean cup lebih baik"),
            "sweetness": (10.0, "Optimalkan waktu panen saat cherry matang sempurna")
        }
        
        for feature, (threshold, rec) in rec_map.items():
            if features.get(feature, 10) < threshold:
                recommendations.append(rec)
        
        # Moisture recommendation
        moisture = features.get("moisture_percentage", 11)
        if moisture < 9:
            recommendations.append(f"Moisture terlalu rendah ({moisture:.1f}%). Ideal: 9-12%")
        elif moisture > 12.5:
            recommendations.append(f"Moisture terlalu tinggi ({moisture:.1f}%). Ideal: 9-12%")
        
        # Grade-specific summary
        if grade == "A":
            recommendations.insert(0, "✓ Kualitas Specialty! Pertahankan standar ini untuk konsistensi.")
        elif grade == "B":
            recommendations.insert(0, "→ Kualitas Premium. Perbaikan minor dapat meningkatkan ke Grade A.")
        else:
            recommendations.insert(0, "⚠ Perlu perbaikan signifikan untuk mencapai standar specialty.")
        
        return recommendations[:6]  # Return top 6 recommendations

    def save(self) -> bool:
        """Save trained model, scaler, and metadata to disk."""
        try:
            os.makedirs(settings.MODEL_DIR, exist_ok=True)
            
            # Save model
            joblib.dump(self.model, settings.MODEL_PATH)
            
            # Save scaler and label encoder
            joblib.dump(self.scaler, settings.SCALER_PATH)
            
            # Save metadata
            metadata = {
                "label_encoder": self.label_encoder,
                "metrics": self.metrics,
                "metadata": self.metadata,
                "training_history": self.training_history,
                "network_architecture": self.network_architecture,
                "is_trained": self.is_trained
            }
            joblib.dump(metadata, settings.METADATA_PATH)
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load(self) -> bool:
        """Load trained model from disk."""
        try:
            if not all(os.path.exists(p) for p in [
                settings.MODEL_PATH, 
                settings.SCALER_PATH, 
                settings.METADATA_PATH
            ]):
                return False
            
            # Load model
            self.model = joblib.load(settings.MODEL_PATH)
            
            # Load scaler
            self.scaler = joblib.load(settings.SCALER_PATH)
            
            # Load metadata
            metadata = joblib.load(settings.METADATA_PATH)
            self.label_encoder = metadata["label_encoder"]
            self.metrics = metadata["metrics"]
            self.metadata = metadata["metadata"]
            self.training_history = metadata["training_history"]
            self.network_architecture = metadata["network_architecture"]
            self.is_trained = metadata["is_trained"]
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_type": "Backpropagation Neural Network (MLP)",
            "algorithm": "Backpropagation with Adam Optimizer",
            "is_trained": self.is_trained,
            "network_architecture": self.network_architecture,
            "training_config": BACKPROPAGATION_CONFIG,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "training_history": {
                "n_iterations": self.training_history.get("n_iterations"),
                "final_loss": self.training_history.get("final_loss"),
                "best_loss": self.training_history.get("best_loss")
            }
        }


# Global model instance
neural_network = BackpropagationNeuralNetwork()
