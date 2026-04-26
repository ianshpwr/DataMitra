from pydantic import BaseModel
from typing  import Optional, Any

class ChartPlan(BaseModel):
    """
    One chart plan per insight.
    The agent decides type + what to show.
    The renderer uses real DataFrame data to build it.
    """
    insight_id:  str
    chart_type:  str          # bar | histogram | line | scatter | heatmap | pie
    title:       str
    subtitle:    str
    x_column:    Optional[str] = None
    y_column:    Optional[str] = None
    color_col:   Optional[str] = None
    aggregation: Optional[str] = None  # count | mean | sum | none
    filters:     dict          = {}
    highlight:   dict          = {}    # {"column": "total_amount", "above": 5000}
    reasoning:   str           = ""    # why this chart type was chosen
    skip:        bool          = False # agent decided no chart needed
    skip_reason: str           = ""