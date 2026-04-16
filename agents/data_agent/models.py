from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime
import polars as pl

class ColumnProfile(BaseModel):
    name:         str
    dtype:        str
    null_pct:     float
    unique_count: int
    sample_vals:  list[Any]
    semantic_tag: Optional[str] = None  # e.g. "currency", "date", "id", "status"
    issues:       list[str]     = []    # e.g. ["mixed_date_formats", "leading_dollar_signs"]

class QualityReport(BaseModel):
    total_rows:      int
    duplicate_rows:  int
    null_row_pct:    float
    columns_failing: list[str]
    overall_score:   float          # 0.0 – 1.0
    passed:          bool

class DataContext(BaseModel):
    """
    The canonical output of the Data Processing Agent.
    Every downstream agent receives exactly this object.
    """
    source_type:    str                  # "static_csv" | "live_stream" | "database" | "api"
    domain:         str                  # "ecommerce" | "finance" | "saas" | "unknown"
    ingested_at:    datetime             = Field(default_factory=datetime.utcnow)
    row_count:      int
    column_count:   int
    columns:        list[ColumnProfile]
    quality:        QualityReport
    processing_ms:  int                  # how long the agent took
    warnings:       list[str]            = []

    # The actual cleaned DataFrame is stored separately (not serialized to JSON)
    # Downstream agents receive it via the agent's run() return value
    class Config:
        arbitrary_types_allowed = True