from pydantic import BaseModel
from typing import Optional
from enum import Enum

class InsightType(str, Enum):
    trend        = "trend"
    anomaly      = "anomaly"
    correlation  = "correlation"
    distribution = "distribution"
    comparison   = "comparison"
    summary      = "summary"

class InsightSeverity(str, Enum):
    info     = "info"      # normal observation
    warning  = "warning"   # needs attention
    critical = "critical"  # needs immediate action

class StatEvidence(BaseModel):
    """Raw numbers that back up an insight — no LLM involved."""
    metric:      str
    value:       float | int | str
    comparison:  Optional[str] = None   # e.g. "vs last 30 days: +12%"
    column:      Optional[str] = None

class Insight(BaseModel):
    id:              str
    type:            InsightType
    severity:        InsightSeverity
    title:           str                # one-line human summary
    explanation:     str                # 2–4 sentence LLM explanation
    evidence:        list[StatEvidence] # hard data points backing it up
    affected_columns: list[str]
    confidence:      float              # 0.0–1.0, set by Critic later
    actionable:      bool               # does this need a decision?
    raw_stats:       dict               # full stat dict for audit

class InsightBundle(BaseModel):
    """Output of the Insight Agent — input to the Critic Agent."""
    domain:         str
    source_type:    str
    total_rows:     int
    insights:       list[Insight]
    executive_summary: str             # 3-sentence overview for the dashboard
    analysis_ms:    int
    llm_model_used: str
    token_count:    int