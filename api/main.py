import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routers.analysis import router as analysis_router
from api.routers.health   import router as health_router

load_dotenv()

app = FastAPI(
    title="DataMitra API",
    description="AI-powered data analysis — Agents 1+2+3 pipeline",
    version="0.1.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",
)

# Allow Streamlit Cloud + local dev to call the API.
# Set ALLOWED_ORIGINS in Render env vars to lock this down in production,
# e.g. "https://your-app.streamlit.app,http://localhost:8501"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_raw_origins != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analysis_router)

@app.get("/")
def root():
    return {"message": "DataMitra API is running", "docs": "/docs"}