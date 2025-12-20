"""
Health check endpoints.
"""

from fastapi import APIRouter, status
from src.schemas.response import HealthResponse
from src.models.backpropagation import neural_network
from src.core.config import settings
from datetime import datetime

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check API and model health status"
)
async def health_check():
    """
    Perform health check on the API service.
    
    Returns:
        - status: Service status (healthy/unhealthy)
        - model_status: Model status (ready/not_ready)
        - version: API version
    """
    return HealthResponse(
        status="healthy",
        model_status="ready" if neural_network.is_trained else "not_ready",
        version=settings.APP_VERSION,
        timestamp=datetime.now()
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if service is ready to accept requests"
)
async def readiness_check():
    """Check if the service is ready to handle requests."""
    if neural_network.is_trained:
        return {
            "ready": True,
            "message": "Service is ready"
        }
    return {
        "ready": False,
        "message": "Model not trained yet"
    }


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Check if service is alive"
)
async def liveness_check():
    """Simple liveness check."""
    return {"alive": True}
