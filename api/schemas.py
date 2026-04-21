from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class EvidenceOut(BaseModel):
    metric:  str
    value:   float | int | str
    column:  Optional[str] = None

class InsightOut(BaseModel):
    id:               str
    type:             str
    severity:         str
    title:            str
    explanation:      str
    evidence:         list[EvidenceOut]
    affected_columns: list[str]
    confidence:       float
    actionable:       bool

class QualityReportOut(BaseModel):
    total_rows:      int
    duplicate_rows:  int
    overall_score:   float
    passed:          bool
    columns_failing: list[str]

class CriticReportOut(BaseModel):
    passed_count:       int
    rejected_count:     int
    flagged_count:      int
    overall_quality:    float
    needs_human_review: bool
    retry_was_used:     bool

class AnalysisResponse(BaseModel):
    success:           bool
    domain:            str
    source_type:       str
    total_rows:        int
    executive_summary: str
    insights:          list[InsightOut]
    quality:           QualityReportOut
    critic:            CriticReportOut
    processing_ms:     int
    token_count:       int
    analysed_at:       datetime
    decisions: Optional[DecisionPackOut] = None 

class ErrorResponse(BaseModel):
    success: bool = False
    error:   str
    detail:  Optional[str] = None

class HealthResponse(BaseModel):
    status:  str
    version: str
    agents:  dict[str, str]