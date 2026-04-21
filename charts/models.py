from pydantic import BaseModel
from typing  import Optional, Any

class ChartAnnotation(BaseModel):
    value:  float | str
    label:  str
    color:  str = "#ef4444"

class ChartSpec(BaseModel):
    """
    A declarative description of one chart.
    Insight/Decision agents produce specs.
    The renderer consumes them.
    This separation means: swap Plotly for ECharts, Vega-Lite,
    or a BI export — only the renderer changes.
    """
    chart_id:    str
    insight_id:  Optional[str]    = None
    decision_id: Optional[str]    = None
    type:        str               # histogram | bar | line | scatter_matrix | box
    intent:      str               # show_anomaly | show_distribution |
                                   # show_trend | show_comparison |
                                   # show_decision_matrix
    title:       str
    subtitle:    str               = ""
    columns:     list[str]         = []
    annotations: list[ChartAnnotation] = []
    meta:        dict[str, Any]    = {}  # extra data the renderer needs