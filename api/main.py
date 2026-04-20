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

# Allow React dev server on port 3000 to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analysis_router)

@app.get("/")
def root():
    return {"message": "DataMitra API is running", "docs": "/docs"}