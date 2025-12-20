"""
Custom exceptions for the Coffee Quality Expert System.
"""

from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Base exception for API errors."""
    
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: dict = None
    ):
        self.error_code = error_code
        self.details = details or {}
        super().__init__(
            status_code=status_code,
            detail={
                "error_code": error_code,
                "message": message,
                "details": self.details
            }
        )


class ModelNotTrainedException(BaseAPIException):
    """Raised when model is not trained yet."""
    
    def __init__(self, message: str = "Model belum ditraining"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MODEL_NOT_TRAINED",
            message=message
        )


class ModelLoadException(BaseAPIException):
    """Raised when model fails to load."""
    
    def __init__(self, message: str = "Gagal memuat model"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="MODEL_LOAD_ERROR",
            message=message
        )


class DatasetNotFoundException(BaseAPIException):
    """Raised when dataset file is not found."""
    
    def __init__(self, path: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="DATASET_NOT_FOUND",
            message=f"Dataset tidak ditemukan: {path}",
            details={"path": path}
        )


class InvalidInputException(BaseAPIException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="INVALID_INPUT",
            message=message,
            details={"field": field} if field else {}
        )


class PredictionException(BaseAPIException):
    """Raised when prediction fails."""
    
    def __init__(self, message: str = "Gagal melakukan prediksi"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="PREDICTION_ERROR",
            message=message
        )


class TrainingException(BaseAPIException):
    """Raised when training fails."""
    
    def __init__(self, message: str = "Gagal melatih model"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="TRAINING_ERROR",
            message=message
        )
