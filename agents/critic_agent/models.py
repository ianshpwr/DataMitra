from pydantic import BaseModel
from typing import Optional
from agents.insight_agent.models import Insight, InsightBundle

class InsightVerdict(BaseModel):
    insight_id:       str
    passed:           bool
    confidence_score: float        # 0.0–1.0
    rejection_reason: Optional[str] = None
    flags:            list[str]    = []  # e.g. ["no_data_citation", "vague_title"]

class CriticReport(BaseModel):
    passed_count:    int
    rejected_count:  int
    flagged_count:   int
    overall_quality: float         # mean confidence of passed insights
    needs_human_review: bool
    verdicts:        list[InsightVerdict]
    retry_recommended: bool
    retry_reason:    Optional[str] = None