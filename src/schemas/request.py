"""
Request schemas for API endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CoffeeQualityInput(BaseModel):
    """
    Input schema for coffee quality prediction.
    
    All scores follow SCA (Specialty Coffee Association) cupping protocol.
    """
    
    aroma: float = Field(
        ...,
        ge=0,
        le=10,
        description="Intensitas dan kualitas aroma kopi (0-10)",
        json_schema_extra={"example": 8.0}
    )
    flavor: float = Field(
        ...,
        ge=0,
        le=10,
        description="Rasa keseluruhan kopi (0-10)",
        json_schema_extra={"example": 8.0}
    )
    aftertaste: float = Field(
        ...,
        ge=0,
        le=10,
        description="Rasa yang tertinggal setelah minum (0-10)",
        json_schema_extra={"example": 7.5}
    )
    acidity: float = Field(
        ...,
        ge=0,
        le=10,
        description="Tingkat keasaman kopi (0-10)",
        json_schema_extra={"example": 8.0}
    )
    body: float = Field(
        ...,
        ge=0,
        le=10,
        description="Ketebalan dan tekstur kopi di mulut (0-10)",
        json_schema_extra={"example": 7.5}
    )
    balance: float = Field(
        ...,
        ge=0,
        le=10,
        description="Keseimbangan antara flavor, aftertaste, acidity, dan body (0-10)",
        json_schema_extra={"example": 8.0}
    )
    uniformity: float = Field(
        ...,
        ge=0,
        le=10,
        description="Konsistensi rasa antar cup (0-10)",
        json_schema_extra={"example": 10.0}
    )
    clean_cup: float = Field(
        ...,
        ge=0,
        le=10,
        description="Kebersihan rasa tanpa defect (0-10)",
        json_schema_extra={"example": 10.0}
    )
    sweetness: float = Field(
        ...,
        ge=0,
        le=10,
        description="Tingkat kemanisan alami kopi (0-10)",
        json_schema_extra={"example": 10.0}
    )
    moisture_percentage: float = Field(
        ...,
        ge=0,
        le=20,
        description="Persentase kelembaban biji kopi (0-20%)",
        json_schema_extra={"example": 11.0}
    )
    
    @field_validator('moisture_percentage')
    @classmethod
    def validate_moisture(cls, v):
        """Validate moisture is within acceptable range."""
        if v < 8 or v > 12.5:
            # Warning but still accept
            pass
        return v
    
    def to_feature_list(self) -> list:
        """Convert to ordered feature list for model input."""
        return [
            self.aroma,
            self.flavor,
            self.aftertaste,
            self.acidity,
            self.body,
            self.balance,
            self.uniformity,
            self.clean_cup,
            self.sweetness,
            self.moisture_percentage
        ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "aroma": self.aroma,
            "flavor": self.flavor,
            "aftertaste": self.aftertaste,
            "acidity": self.acidity,
            "body": self.body,
            "balance": self.balance,
            "uniformity": self.uniformity,
            "clean_cup": self.clean_cup,
            "sweetness": self.sweetness,
            "moisture_percentage": self.moisture_percentage
        }


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction."""
    
    samples: list[CoffeeQualityInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of coffee samples to predict (max 100)"
    )
