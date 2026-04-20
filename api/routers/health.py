from fastapi import APIRouter
from api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        agents={
            "data_agent":    "ready",
            "insight_agent": "ready",
            "critic_agent":  "ready",
        },
    )