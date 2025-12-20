"""
Reference data endpoints for frontend integration.
"""

from fastapi import APIRouter, status
from datetime import datetime

from src.schemas.response import GradeInfoResponse, FeatureInfoResponse
from src.core.config import GRADE_THRESHOLDS, FEATURE_COLUMNS

router = APIRouter(prefix="/reference", tags=["Reference Data"])


@router.get(
    "/grades",
    response_model=GradeInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Grade Information",
    description="Get coffee quality grade classification information"
)
async def get_grades():
    """
    Get information about coffee quality grade classifications.
    
    Based on SCA (Specialty Coffee Association) standards:
    - Grade A: Specialty Grade (≥85 points)
    - Grade B: Premium Grade (80-84.99 points)
    - Grade C: Below Premium (<80 points)
    """
    grades = []
    for grade, info in GRADE_THRESHOLDS.items():
        grades.append({
            "grade": grade,
            "min_score": info["min"],
            "max_score": info["max"],
            "label": info["label"],
            "description": info["description"],
            "color": {
                "A": "#22c55e",  # Green
                "B": "#eab308",  # Yellow
                "C": "#ef4444"   # Red
            }.get(grade, "#6b7280")
        })
    
    return GradeInfoResponse(
        success=True,
        message="Grade information retrieved",
        timestamp=datetime.now(),
        data={
            "grades": grades,
            "scoring_system": "SCA (Specialty Coffee Association) Cupping Protocol",
            "total_possible_points": 100
        }
    )


@router.get(
    "/features",
    response_model=FeatureInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Feature Information",
    description="Get information about input features for prediction"
)
async def get_features():
    """
    Get detailed information about input features.
    
    All sensory features follow SCA cupping protocol scoring.
    """
    features = [
        {
            "name": "aroma",
            "display_name": "Aroma",
            "description": "Intensitas dan kualitas aroma kopi, dinilai saat dry dan wet",
            "min": 0,
            "max": 10,
            "step": 0.25,
            "category": "sensory",
            "weight": "high",
            "tips": "Nilai aroma dari fragrance (kering) dan aroma (basah)"
        },
        {
            "name": "flavor",
            "display_name": "Flavor",
            "description": "Rasa keseluruhan yang mencakup taste dan aroma retro-nasal",
            "min": 0,
            "max": 10,
            "step": 0.25,
            "category": "sensory",
            "weight": "high",
            "tips": "Kombinasi taste buds dan aroma yang naik ke hidung"
        },
        {
            "name": "aftertaste",
            "display_name": "Aftertaste",
            "description": "Rasa yang tertinggal setelah kopi ditelan atau dibuang",
            "min": 0,
            "max": 10,
            "step": 0.25,
            "category": "sensory",
            "weight": "medium",
            "tips": "Durasi dan kualitas rasa yang bertahan"
        },
        {
            "name": "acidity",
            "display_name": "Acidity",
            "description": "Tingkat keasaman yang memberikan brightness pada kopi",
            "min": 0,
            "max": 10,
            "step": 0.25,
            "category": "sensory",
            "weight": "high",
            "tips": "Acidity yang baik terasa bright, bukan sour"
        },
        {
            "name": "body",
            "display_name": "Body",
            "description": "Ketebalan dan tekstur kopi di mulut (mouthfeel)",
            "min": 0,
            "max": 10,
            "step": 0.25,
            "category": "sensory",
            "weight": "medium",
            "tips": "Sensasi fisik kopi di lidah dan langit-langit"
        },
        {
            "name": "balance",
            "display_name": "Balance",
            "description": "Keseimbangan antara flavor, aftertaste, acidity, dan body",
            "min": 0,
            "max": 10,
            "step": 0.25,
            "category": "sensory",
            "weight": "high",
            "tips": "Tidak ada atribut yang mendominasi secara negatif"
        },
        {
            "name": "uniformity",
            "display_name": "Uniformity",
            "description": "Konsistensi rasa antar cup dalam satu sampel",
            "min": 0,
            "max": 10,
            "step": 2,
            "category": "quality",
            "weight": "medium",
            "tips": "2 poin per cup yang konsisten (5 cups = 10 poin)"
        },
        {
            "name": "clean_cup",
            "display_name": "Clean Cup",
            "description": "Kebersihan rasa tanpa defect atau off-flavors",
            "min": 0,
            "max": 10,
            "step": 2,
            "category": "quality",
            "weight": "medium",
            "tips": "2 poin per cup yang bersih (5 cups = 10 poin)"
        },
        {
            "name": "sweetness",
            "display_name": "Sweetness",
            "description": "Tingkat kemanisan alami dari kopi",
            "min": 0,
            "max": 10,
            "step": 2,
            "category": "quality",
            "weight": "medium",
            "tips": "2 poin per cup dengan sweetness (5 cups = 10 poin)"
        },
        {
            "name": "moisture_percentage",
            "display_name": "Moisture",
            "description": "Persentase kelembaban biji kopi hijau",
            "min": 0,
            "max": 20,
            "step": 0.1,
            "category": "physical",
            "weight": "low",
            "tips": "Ideal: 9-12%. Terlalu rendah = stale, terlalu tinggi = risiko jamur"
        }
    ]
    
    return FeatureInfoResponse(
        success=True,
        message="Feature information retrieved",
        timestamp=datetime.now(),
        data={
            "features": features,
            "total_features": len(features),
            "categories": {
                "sensory": "Atribut sensorik yang dinilai melalui cupping",
                "quality": "Atribut kualitas berdasarkan konsistensi",
                "physical": "Karakteristik fisik biji kopi"
            }
        }
    )


@router.get(
    "/cupping-guide",
    status_code=status.HTTP_200_OK,
    summary="Get Cupping Guide",
    description="Get SCA cupping protocol guide"
)
async def get_cupping_guide():
    """Get SCA cupping protocol guide for reference."""
    return {
        "success": True,
        "message": "Cupping guide retrieved",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "protocol": "SCA Cupping Protocol",
            "steps": [
                {
                    "step": 1,
                    "name": "Fragrance/Aroma",
                    "description": "Evaluasi aroma kopi kering dan setelah ditambah air panas"
                },
                {
                    "step": 2,
                    "name": "Breaking the Crust",
                    "description": "Pecahkan kerak dan evaluasi aroma yang keluar"
                },
                {
                    "step": 3,
                    "name": "Tasting",
                    "description": "Slurp kopi untuk mengevaluasi flavor, aftertaste, acidity, body, balance"
                },
                {
                    "step": 4,
                    "name": "Scoring",
                    "description": "Berikan skor untuk setiap atribut (6-10 untuk specialty)"
                }
            ],
            "scoring_scale": {
                "6.00-6.75": "Good",
                "7.00-7.75": "Very Good",
                "8.00-8.75": "Excellent",
                "9.00-9.75": "Outstanding"
            },
            "specialty_threshold": 80,
            "reference_url": "https://sca.coffee/research/protocols-best-practices"
        }
    }
