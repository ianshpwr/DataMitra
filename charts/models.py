from pydantic import BaseModel
from typing  import Optional, Any

class ChartPlan(BaseModel):
    insight_id:  str
    chart_type:  str
    title:       str
    subtitle:    str          = ""
    x_column:    Optional[str] = None
    y_column:    Optional[str] = None
    color_col:   Optional[str] = None
    aggregation: Optional[str] = None
    filters:     dict          = {}
    highlight:   dict          = {}
    reasoning:   str           = ""
    skip:        bool          = False
    skip_reason: str           = ""
