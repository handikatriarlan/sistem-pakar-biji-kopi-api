"""Neural Network model for coffee quality prediction using Backpropagation."""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import os
from typing import Tuple, Dict, Any

from app.config import FEATURE_COLUMNS, GRADE_THRESHOLDS, NN_CONFIG, MODEL_PATH, SCALER_PATH


class CoffeeQualityModel:
    """
    Coffee Quality Expert System using Backpropagation Neural Network.
    
    This model predicts coffee quality grade (A, B, C) based on
    sensory evaluation features using a Multi-Layer Perceptron.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.metrics = {}
        self.grade_distribution = {}
        self.training_samples = 0
        
    def _assign_grade(self, total_cup_points: float) -> str:
        """Assign quality grade based on total cup points."""
        if total_cup_points >= GRADE_THRESHOLDS["A"]:
            return "A"
        elif total_cup_points >= GRADE_THRESHOLDS["B"]:
            return "B"
        else:
            return "C"
    
    def _get_quality_label(self, grade: str) -> str:
        """Get descriptive quality label for grade."""
        labels = {
            "A": "Specialty Grade - Excellent Quality",
            "B": "Premium Grade - Good Quality", 
            "C": "Standard Grade - Below Premium"
        }
        return labels.get(grade, "Unknown")
    
    def _generate_recommendations(self, features: Dict[str, float], grade: str) -> list:
        """Generate quality improvement recommendations based on features."""
        recommendations = []
        
        # Analyze each feature and provide recommendations
        if features.get("aroma", 10) < 7.5:
            recommendations.append("Tingkatkan proses pengeringan untuk aroma yang lebih baik")
        
        if features.get("flavor", 10) < 7.5:
            recommendations.append("Perhatikan proses roasting untuk meningkatkan flavor")
            
        if features.get("aftertaste", 10) < 7.5:
            recommendations.append("Evaluasi kualitas biji mentah untuk aftertaste yang lebih baik")
            
        if features.get("acidity", 10) < 7.5:
            recommendations.append("Sesuaikan ketinggian tanam atau varietas untuk acidity optimal")
            
        if features.get("body", 10) < 7.5:
            recommendations.append("Pertimbangkan metode processing yang berbeda untuk body lebih kuat")
            
        if features.get("balance", 10) < 7.5:
            recommendations.append("Fokus pada keseimbangan profil rasa secara keseluruhan")
            
        if features.get("uniformity", 10) < 10:
            recommendations.append("Tingkatkan konsistensi dalam proses sorting dan grading")
            
        if features.get("clean_cup", 10) < 10:
            recommendations.append("Perbaiki proses pencucian dan fermentasi")
            
        if features.get("sweetness", 10) < 10:
            recommendations.append("Optimalkan waktu panen untuk sweetness maksimal")
            
        moisture = features.get("moisture_percentage", 11)
        if moisture < 9 or moisture > 12:
            recommendations.append(f"Sesuaikan moisture content (ideal: 9-12%, saat ini: {moisture:.1f}%)")
        
        if grade == "A":
            recommendations.insert(0, "Kualitas sudah excellent! Pertahankan standar ini.")
        elif grade == "B":
            recommendations.insert(0, "Kualitas baik, beberapa perbaikan dapat meningkatkan ke Grade A.")
        else:
            recommendations.insert(0, "Perlu perbaikan signifikan untuk mencapai standar specialty.")
            
        return recommendations[:5]  # Return top 5 recommendations

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data for neural network training.
        
        Steps:
        1. Select relevant features
        2. Handle missing values
        3. Create labels based on Total Cup Points
        4. Normalize features using StandardScaler
        """
        # Select features
        feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Create grade labels
        y_grades = df["Total Cup Points"].apply(self._assign_grade)
        y_scores = df["Total Cup Points"].values
        
        # Store grade distribution
        self.grade_distribution = y_grades.value_counts().to_dict()
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_grades)
        
        return X_scaled, y_encoded, y_scores
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the Backpropagation Neural Network model.
        
        Uses MLPClassifier with backpropagation algorithm for
        multi-class classification (Grade A, B, C).
        """
        # Preprocess data
        X, y, y_scores = self.preprocess_data(df)
        self.training_samples = len(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train model
        self.model = MLPClassifier(
            hidden_layer_sizes=NN_CONFIG["hidden_layers"],
            max_iter=NN_CONFIG["max_iter"],
            learning_rate_init=NN_CONFIG["learning_rate_init"],
            activation=NN_CONFIG["activation"],
            solver=NN_CONFIG["solver"],
            random_state=NN_CONFIG["random_state"],
            early_stopping=NN_CONFIG["early_stopping"],
            validation_fraction=NN_CONFIG["validation_fraction"],
            verbose=False
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.metrics = {
            "accuracy": float(accuracy),
            "training_loss": float(self.model.loss_),
            "n_iterations": self.model.n_iter_
        }
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict coffee quality grade from input features.
        
        Returns grade, predicted score, confidence, and recommendations.
        """
        if not self.is_trained:
            raise ValueError("Model belum ditraining. Jalankan train() terlebih dahulu.")
        
        # Prepare input
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
        
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        grade_encoded = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        grade = self.label_encoder.inverse_transform([grade_encoded])[0]
        confidence = float(max(probabilities))
        
        # Estimate total cup points from features
        base_score = sum(feature_values[:9])  # Sum of sensory scores
        predicted_score = min(100, max(0, base_score))
        
        return {
            "grade": grade,
            "predicted_score": round(predicted_score, 2),
            "confidence": round(confidence, 4),
            "quality_label": self._get_quality_label(grade),
            "recommendations": self._generate_recommendations(features, grade)
        }
    
    def save_model(self):
        """Save trained model and scaler to disk."""
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump({
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "metrics": self.metrics,
            "grade_distribution": self.grade_distribution,
            "training_samples": self.training_samples
        }, SCALER_PATH)
    
    def load_model(self) -> bool:
        """Load trained model from disk."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = joblib.load(MODEL_PATH)
                data = joblib.load(SCALER_PATH)
                self.scaler = data["scaler"]
                self.label_encoder = data["label_encoder"]
                self.metrics = data["metrics"]
                self.grade_distribution = data["grade_distribution"]
                self.training_samples = data["training_samples"]
                self.is_trained = True
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False


# Global model instance
coffee_model = CoffeeQualityModel()
