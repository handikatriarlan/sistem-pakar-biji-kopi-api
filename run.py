"""
Application entry point.

Run the Coffee Quality Expert System API server.
"""

import uvicorn
from src.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
