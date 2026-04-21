"""
Builds ChartSpecs from InsightBundle + DecisionPack.
One insight → zero or one spec (some insight types don't need charts).
Decision matrix → one spec covering all decisions.
"""
from .models import ChartSpec, ChartAnnotation
from agents.insight_agent.models import Insight, InsightBundle, InsightType
from agents.decision_agent.models import DecisionPack


def build_specs(
    bundle:   InsightBundle,
    pack:     DecisionPack | None,
    raw_data: dict,            # the full API response dict
) -> list[ChartSpec]:
    specs: list[ChartSpec] = []
    chart_counter = 0

    def _id():
        nonlocal chart_counter
        chart_counter += 1
        return f"CHT-{chart_counter:03d}"

    # ── One spec per insight ──────────────────────────────────────────────────
    for ins in bundle.insights:
        spec = _spec_for_insight(ins, raw_data, _id())
        if spec:
            specs.append(spec)

    # ── One decision matrix if we have 2+ decisions ───────────────────────────
    if pack and len(pack.decisions) >= 2:
        specs.append(_decision_matrix_spec(pack, _id()))

    return specs


# ── Per-insight spec routing ──────────────────────────────────────────────────

def _spec_for_insight(ins: Insight, raw_data: dict, cid: str) -> ChartSpec | None:
    t = ins.type

    if t == InsightType.anomaly:
        return _anomaly_spec(ins, raw_data, cid)

    if t == InsightType.distribution:
        # Categorical vs numeric — detected by presence of top_5 evidence
        has_top5 = any(e.metric == "top_5" for e in ins.evidence)
        if has_top5:
            return _categorical_bar_spec(ins, cid)
        else:
            return _numeric_spread_spec(ins, cid)

    if t == InsightType.trend:
        return _trend_spec(ins, raw_data, cid)

    if t == InsightType.correlation:
        return _correlation_spec(ins, cid)

    return None   # summary insights don't need a chart


# ── Spec factories ────────────────────────────────────────────────────────────

def _anomaly_spec(ins: Insight, raw_data: dict, cid: str) -> ChartSpec:
    col = ins.affected_columns[0] if ins.affected_columns else "value"

    # Collect outlier values from evidence for annotations
    annotations = []
    for ev in ins.evidence:
        if ev.metric == "anomaly_values" and isinstance(ev.value, list):
            for v in ev.value[:3]:   # annotate up to 3 outliers
                try:
                    annotations.append(ChartAnnotation(
                        value=float(v),
                        label=f"Outlier: {float(v):,.0f}",
                        color="#ef4444",
                    ))
                except (ValueError, TypeError):
                    pass

    return ChartSpec(
        chart_id=cid,
        insight_id=ins.id,
        type="histogram",
        intent="show_anomaly",
        title=f"{col.replace('_',' ').title()} — anomaly in context",
        subtitle=f"{ins.title} · Red lines mark detected outliers",
        columns=[col],
        annotations=annotations,
        meta={
            "mean":     _get_ev(ins, "column_mean"),
            "std":      _get_ev(ins, "column_std"),
            "n_anomalies": _get_ev(ins, "n_anomalies"),
        },
    )


def _categorical_bar_spec(ins: Insight, cid: str) -> ChartSpec:
    col   = ins.affected_columns[0] if ins.affected_columns else "category"
    top5  = {}
    top_pct = 0.0

    for ev in ins.evidence:
        if ev.metric == "top_5":
            try:
                import json
                top5 = (json.loads(ev.value)
                        if isinstance(ev.value, str)
                        else ev.value) or {}
            except Exception:
                pass
        if ev.metric == "top_pct":
            try:
                top_pct = float(ev.value)
            except (ValueError, TypeError):
                pass

    return ChartSpec(
        chart_id=cid,
        insight_id=ins.id,
        type="bar",
        intent="show_distribution",
        title=f"{col.replace('_',' ').title()} breakdown",
        subtitle=f"Top value accounts for {top_pct*100:.1f}% of records",
        columns=[col],
        meta={"top5": top5, "top_pct": top_pct},
    )


def _numeric_spread_spec(ins: Insight, cid: str) -> ChartSpec:
    col = ins.affected_columns[0] if ins.affected_columns else "value"

    return ChartSpec(
        chart_id=cid,
        insight_id=ins.id,
        type="box",
        intent="show_comparison",
        title=f"{col.replace('_',' ').title()} — spread and outliers",
        subtitle="Box = IQR · Diamond = mean · Dots = outliers",
        columns=[col],
        meta={
            "mean":         _get_ev(ins, "mean"),
            "std":          _get_ev(ins, "std"),
            "q1":           _get_ev(ins, "q1"),
            "q3":           _get_ev(ins, "q3"),
            "iqr":          _get_ev(ins, "iqr"),
            "outlier_pct":  _get_ev(ins, "outlier_pct"),
            "skew":         _get_ev(ins, "skew"),
        },
    )


def _trend_spec(ins: Insight, raw_data: dict, cid: str) -> ChartSpec:
    monthly = []
    direction = "up"
    trend_pct = 0.0

    for ev in ins.evidence:
        if ev.metric == "monthly_counts" and isinstance(ev.value, list):
            monthly = ev.value
        if ev.metric == "direction":
            direction = str(ev.value)
        if ev.metric == "trend_pct_per_period":
            try:
                trend_pct = float(ev.value)
            except (ValueError, TypeError):
                pass

    return ChartSpec(
        chart_id=cid,
        insight_id=ins.id,
        type="line",
        intent="show_trend",
        title="Order volume trend over time",
        subtitle=f"{'↑' if direction == 'up' else '↓'} {abs(trend_pct):.1f}% per period · "
                 f"{'Growing' if direction == 'up' else 'Declining'} trend",
        columns=ins.affected_columns,
        meta={
            "monthly_counts": monthly,
            "direction":      direction,
            "trend_pct":      trend_pct,
            "r_squared":      _get_ev(ins, "r_squared"),
        },
    )


def _correlation_spec(ins: Insight, cid: str) -> ChartSpec:
    r     = _get_ev(ins, "pearson_r")
    p_val = _get_ev(ins, "p_value")
    cols  = ins.affected_columns[:2] if len(ins.affected_columns) >= 2 else ins.affected_columns

    return ChartSpec(
        chart_id=cid,
        insight_id=ins.id,
        type="scatter_matrix",
        intent="show_correlation",
        title=f"Correlation: {' vs '.join(c.replace('_',' ') for c in cols)}",
        subtitle=f"Pearson r = {r:.3f} · p = {p_val:.4f} · "
                 f"{'Strong' if abs(r) > 0.7 else 'Moderate'} "
                 f"{'positive' if r > 0 else 'negative'} correlation",
        columns=cols,
        meta={"pearson_r": r, "p_value": p_val},
    )


def _decision_matrix_spec(pack: DecisionPack, cid: str) -> ChartSpec:
    EFFORT_MAP = {"low": 1, "medium": 2, "high": 3}
    IMPACT_MAP = {"low": 1, "medium": 2, "high": 3}

    decisions_data = [
        {
            "id":           d.id,
            "title":        d.title[:40],
            "effort":       EFFORT_MAP.get(d.effort_level, 2),
            "impact":       IMPACT_MAP.get(d.impact_level, 2),
            "priority":     d.priority_score,
            "action_type":  d.action_type,
            "owner":        d.owner or "—",
        }
        for d in pack.decisions
    ]

    return ChartSpec(
        chart_id=cid,
        insight_id=None,
        decision_id="ALL",
        type="scatter_matrix",
        intent="show_decision_matrix",
        title="Decision impact vs effort matrix",
        subtitle="Top-left = quick wins · Top-right = big bets · Bubble size = priority",
        columns=[],
        meta={"decisions": decisions_data},
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_ev(ins: Insight, key: str) -> float:
    for ev in ins.evidence:
        if ev.metric == key:
            try:
                return float(str(ev.value).replace("$","").replace(",",""))
            except (ValueError, TypeError):
                pass
    return 0.0