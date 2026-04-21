from pydantic import BaseModel
from typing import Optional
from enum import Enum

class ActionType(str, Enum):
    investigate  = "investigate"   # look into this further
    fix          = "fix"           # correct a data/process problem
    optimise     = "optimise"      # improve something already working
    monitor      = "monitor"       # watch this metric closely
    escalate     = "escalate"      # needs senior decision-maker

class ImpactLevel(str, Enum):
    high   = "high"
    medium = "medium"
    low    = "low"

class Decision(BaseModel):
    id:              str
    insight_id:      str            # which insight triggered this
    action_type:     ActionType
    title:           str            # one-line action summary
    what:            str            # what exactly to do
    why:             str            # why this action, linked to the insight
    expected_impact: str            # what outcome to expect
    impact_level:    ImpactLevel
    effort_level:    ImpactLevel    # low/medium/high effort to execute
    priority_score:  float          # 0.0–1.0, higher = do first
    risk_if_ignored: str            # consequence of doing nothing
    owner:           Optional[str]  # suggested team/role to own this
    kpi:             Optional[str]  # metric to track success

class DecisionPack(BaseModel):
    """Output of the Decision Agent — one pack per analysis run."""
    domain:          str
    total_insights:  int
    decisions:       list[Decision]
    top_priority:    Optional[Decision]   # highest priority_score
    quick_wins:      list[Decision]       # high impact + low effort
    summary:         str                  # 2-sentence action summary
    generation_ms:   int
    token_count:     int