"""
Renders ChartSpec → Plotly Figure.
All chart logic lives here. Spec builders never touch Plotly.

Extension points:
  - Add a new intent: add one elif branch in render()
  - Export to Vega-Lite: translate spec in a separate exporter
  - NL query: generate a ChartSpec from text, call render()
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .models import ChartSpec

# ── Theme ─────────────────────────────────────────────────────────────────────
DARK_BG    = "#0f1117"
CARD_BG    = "#1e2130"
BORDER     = "#2d3150"
TEXT_MAIN  = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
GRID       = "#252b3b"

ACTION_COLORS = {
    "investigate": "#60a5fa",
    "fix":         "#f87171",
    "optimise":    "#34d399",
    "monitor":     "#a78bfa",
    "escalate":    "#fb923c",
}
PALETTE = ["#60a5fa","#34d399","#f59e0b","#f87171","#a78bfa","#fb923c"]

def _base(title="", subtitle="", height=360):
    annotations = []
    if subtitle:
        annotations.append(dict(
            text=subtitle,
            xref="paper", yref="paper",
            x=0, y=1.06,
            showarrow=False,
            font=dict(color=TEXT_MUTED, size=11),
            align="left",
        ))
    return dict(
        title=dict(
            text=title,
            font=dict(color=TEXT_MAIN, size=15, weight="bold"),
            x=0,
        ),
        annotations=annotations,
        height=height,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_MUTED, size=12),
        margin=dict(l=16, r=16, t=64, b=16),
        legend=dict(
            bgcolor=CARD_BG,
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=TEXT_MUTED, size=11),
        ),
    )

def _ax(reversed=False, suffix="", prefix=""):
    d = dict(
        gridcolor=GRID,
        linecolor=BORDER,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_MUTED),
        zerolinecolor=BORDER,
    )
    if reversed:
        d["autorange"] = "reversed"
    if suffix:
        d["ticksuffix"] = suffix
    if prefix:
        d["tickprefix"] = prefix
    return d


def render(spec: ChartSpec) -> go.Figure | None:
    intent = spec.intent
    try:
        if intent == "show_anomaly":
            return _render_anomaly(spec)
        if intent == "show_distribution":
            return _render_categorical_bar(spec)
        if intent == "show_comparison":
            return _render_numeric_box(spec)
        if intent == "show_trend":
            return _render_trend(spec)
        if intent == "show_correlation":
            return _render_correlation(spec)
        if intent == "show_decision_matrix":
            return _render_decision_matrix(spec)
    except Exception as e:
        print(f"[Renderer] Failed to render {spec.chart_id} ({intent}): {e}")
    return None


# ── Chart 1: Anomaly histogram ────────────────────────────────────────────────

def _render_anomaly(spec: ChartSpec) -> go.Figure:
    col  = spec.columns[0] if spec.columns else "value"
    mean = spec.meta.get("mean", 0)
    std  = spec.meta.get("std", 1) or 1

    # Simulate distribution from mean/std (we don't have raw data in renderer)
    # This renders the theoretical normal + outlier markers
    x = np.linspace(mean - 4*std, mean + 4*std, 300)
    y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean)/std)**2)

    # Normalise to counts scale
    n_anomalies = int(spec.meta.get("n_anomalies", 0))
    scale = max(n_anomalies * 50, 100)
    y_scaled = y * scale

    fig = go.Figure()

    # Normal distribution fill
    fig.add_trace(go.Scatter(
        x=x, y=y_scaled,
        fill="tozeroy",
        fillcolor="rgba(96,165,250,0.15)",
        line=dict(color="#60a5fa", width=1.5),
        name="Expected distribution",
        hovertemplate="Value: %{x:,.2f}<extra></extra>",
    ))

    # Mean line
    fig.add_vline(
        x=mean,
        line=dict(color="#34d399", width=1.5, dash="dash"),
        annotation=dict(
            text=f"Mean: {mean:,.2f}",
            font=dict(color="#34d399", size=11),
            bgcolor=CARD_BG,
        ),
    )

    # +3σ threshold line
    threshold = mean + 3*std
    fig.add_vline(
        x=threshold,
        line=dict(color="#f59e0b", width=1, dash="dot"),
        annotation=dict(
            text="3σ threshold",
            font=dict(color="#f59e0b", size=10),
            bgcolor=CARD_BG,
        ),
    )

    # Outlier markers
    for ann in spec.annotations:
        try:
            v = float(ann.value)
            fig.add_vline(
                x=v,
                line=dict(color="#ef4444", width=1.5),
            )
        except (ValueError, TypeError):
            pass

    # Outlier zone shading
    fig.add_vrect(
        x0=threshold,
        x1=mean + 5*std,
        fillcolor="rgba(239,68,68,0.08)",
        line_width=0,
        annotation=dict(
            text=f"{n_anomalies} outliers",
            font=dict(color="#ef4444", size=11),
            bgcolor=CARD_BG,
        ),
        annotation_position="top left",
    )

    fig.update_layout(
        **_base(spec.title, spec.subtitle, height=340),
        xaxis=dict(**_ax(), title=col.replace("_"," ").title()),
        yaxis=dict(**_ax(), title="Frequency"),
        showlegend=True,
    )
    return fig


# ── Chart 2: Categorical bar ──────────────────────────────────────────────────

def _render_categorical_bar(spec: ChartSpec) -> go.Figure:
    top5 = spec.meta.get("top5", {})
    if not top5:
        return None

    items  = sorted(top5.items(), key=lambda x: x[1], reverse=True)
    labels = [str(k) for k, _ in items]
    values = [int(v) for _, v in items]
    total  = sum(values) or 1
    pcts   = [round(v/total*100, 1) for v in values]
    colors = PALETTE[:len(labels)]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=DARK_BG, width=1),
        ),
        text=[f"{v:,}  ({p}%)" for v, p in zip(values, pcts)],
        textposition="outside",
        textfont=dict(color=TEXT_MAIN, size=11),
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
    ))

    fig.update_layout(
        **_base(spec.title, spec.subtitle, height=max(260, len(labels) * 44)),
        bargap=0.3,
        xaxis=dict(**_ax(), title="Count"),
        yaxis=dict(**_ax(reversed=True)),
    )
    return fig


# ── Chart 3: Numeric box / spread ─────────────────────────────────────────────

def _render_numeric_box(spec: ChartSpec) -> go.Figure:
    col  = spec.columns[0] if spec.columns else "value"
    mean = spec.meta.get("mean", 0)
    std  = spec.meta.get("std", 1) or 1
    q1   = spec.meta.get("q1", mean - std)
    q3   = spec.meta.get("q3", mean + std)
    iqr  = spec.meta.get("iqr", q3 - q1) or 1
    skew = spec.meta.get("skew", 0)

    # Simulate box from stats
    whisker_low  = max(q1 - 1.5*iqr, mean - 3*std)
    whisker_high = min(q3 + 1.5*iqr, mean + 3*std)
    outlier_high = mean + 3*std
    outlier_pct  = spec.meta.get("outlier_pct", 0)

    fig = go.Figure()

    # IQR box
    fig.add_trace(go.Bar(
        x=[col.replace("_"," ")],
        y=[q3 - q1],
        base=[q1],
        width=0.3,
        marker=dict(color="rgba(96,165,250,0.6)",
                    line=dict(color="#60a5fa", width=1.5)),
        name="IQR (25th–75th percentile)",
        hovertemplate=(
            f"Q1: {q1:,.2f}<br>"
            f"Q3: {q3:,.2f}<br>"
            f"IQR: {iqr:,.2f}<extra></extra>"
        ),
    ))

    # Whiskers
    fig.add_trace(go.Scatter(
        x=[col.replace("_"," "), col.replace("_"," ")],
        y=[whisker_low, whisker_high],
        mode="markers",
        marker=dict(symbol="line-ew", size=16,
                    color="#60a5fa",
                    line=dict(color="#60a5fa", width=2)),
        name="Whiskers",
        showlegend=False,
    ))

    # Median line inside box
    median = (q1 + q3) / 2
    fig.add_trace(go.Scatter(
        x=[col.replace("_"," ")],
        y=[median],
        mode="markers",
        marker=dict(symbol="line-ew-open", size=20,
                    color="#f59e0b",
                    line=dict(color="#f59e0b", width=2.5)),
        name=f"Median: {median:,.2f}",
    ))

    # Mean diamond
    fig.add_trace(go.Scatter(
        x=[col.replace("_"," ")],
        y=[mean],
        mode="markers",
        marker=dict(symbol="diamond", size=12,
                    color="white",
                    line=dict(color="#34d399", width=2)),
        name=f"Mean: {mean:,.2f}",
    ))

    # Outlier zone
    if outlier_pct > 0:
        fig.add_hrect(
            y0=outlier_high,
            y1=outlier_high * 1.5,
            fillcolor="rgba(239,68,68,0.07)",
            line_width=0,
        )
        fig.add_hline(
            y=outlier_high,
            line=dict(color="#ef4444", width=1, dash="dot"),
            annotation=dict(
                text=f"Outlier threshold · {outlier_pct*100:.1f}% of values above",
                font=dict(color="#ef4444", size=10),
                bgcolor=CARD_BG,
                xanchor="left",
            ),
            annotation_position="right",
        )

    # Skew annotation
    skew_dir = "right-skewed (long tail of high values)" if skew > 0.5 \
               else "left-skewed (long tail of low values)" if skew < -0.5 \
               else "roughly symmetric"
    fig.add_annotation(
        text=f"Skew: {skew:.2f} — {skew_dir}",
        xref="paper", yref="paper",
        x=1, y=0,
        showarrow=False,
        font=dict(color=TEXT_MUTED, size=10),
        align="right",
    )

    fig.update_layout(
        **_base(spec.title, spec.subtitle, height=400),
        xaxis=dict(**_ax()),
        yaxis=dict(**_ax(), title=col.replace("_"," ").title()),
        showlegend=True,
    )
    return fig


# ── Chart 4: Trend line ───────────────────────────────────────────────────────

def _render_trend(spec: ChartSpec) -> go.Figure:
    monthly = spec.meta.get("monthly_counts", [])
    if not monthly or len(monthly) < 2:
        return None

    direction = spec.meta.get("direction", "up")
    trend_pct = spec.meta.get("trend_pct", 0)
    r2        = spec.meta.get("r_squared", 0)

    x      = list(range(len(monthly)))
    labels = [f"Period {i+1}" for i in x]

    # Linear regression line
    x_arr   = np.array(x, dtype=float)
    y_arr   = np.array(monthly, dtype=float)
    slope   = np.polyfit(x_arr, y_arr, 1)
    trend_y = np.polyval(slope, x_arr).tolist()

    line_color = "#34d399" if direction == "up" else "#ef4444"

    fig = go.Figure()

    # Shaded area under actual
    fig.add_trace(go.Scatter(
        x=labels, y=monthly,
        fill="tozeroy",
        fillcolor="rgba(96,165,250,0.1)",
        line=dict(color="#60a5fa", width=2),
        mode="lines+markers",
        marker=dict(size=6, color="#60a5fa",
                    line=dict(color=CARD_BG, width=1)),
        name="Actual volume",
        hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>",
    ))

    # Trend line
    fig.add_trace(go.Scatter(
        x=labels, y=trend_y,
        line=dict(color=line_color, width=2, dash="dash"),
        mode="lines",
        name=f"Trend ({'+' if trend_pct > 0 else ''}{trend_pct:.1f}%/period)",
        hovertemplate="Trend: %{y:,.0f}<extra></extra>",
    ))

    # R² annotation
    fig.add_annotation(
        text=f"R² = {r2:.3f}  ({'strong' if r2 > 0.7 else 'moderate' if r2 > 0.4 else 'weak'} fit)",
        xref="paper", yref="paper",
        x=1, y=1.02,
        showarrow=False,
        font=dict(color=TEXT_MUTED, size=10),
        align="right",
    )

    fig.update_layout(
        **_base(spec.title, spec.subtitle, height=340),
        xaxis=dict(**_ax(), title="Time period"),
        yaxis=dict(**_ax(), title="Order volume"),
    )
    return fig


# ── Chart 5: Decision impact vs effort matrix ─────────────────────────────────

def _render_decision_matrix(spec: ChartSpec) -> go.Figure:
    decisions = spec.meta.get("decisions", [])
    if not decisions:
        return None

    fig = go.Figure()

    # Quadrant shading
    quadrants = [
        (0.5, 2.5, 1.5, 3.5, "rgba(52,211,153,0.05)", "Quick wins"),
        (1.5, 2.5, 2.5, 3.5, "rgba(251,146,60,0.05)", "Big bets"),
        (0.5, 0.5, 1.5, 2.5, "rgba(96,165,250,0.05)", "Fill-ins"),
        (1.5, 0.5, 2.5, 2.5, "rgba(239,68,68,0.05)",  "Thankless"),
    ]
    quad_labels = [
        (1.0, 3.0, "Quick wins"),
        (2.0, 3.0, "Big bets"),
        (1.0, 1.0, "Fill-ins"),
        (2.0, 1.0, "Thankless"),
    ]
    for x0, y0, x1, y1, color, _ in quadrants:
        fig.add_hrect(y0=y0, y1=y1, x0=x0-0.5, x1=x1-0.5,
                      fillcolor=color, line_width=0)
    for qx, qy, qlabel in quad_labels:
        fig.add_annotation(
            x=qx, y=qy, text=qlabel,
            showarrow=False,
            font=dict(color=TEXT_MUTED, size=10),
            opacity=0.5,
        )

    # Decision bubbles
    for d in decisions:
        color = ACTION_COLORS.get(
            d["action_type"] if isinstance(d["action_type"], str)
            else d["action_type"].value,
            "#6b7280"
        )
        fig.add_trace(go.Scatter(
            x=[d["effort"]],
            y=[d["impact"]],
            mode="markers+text",
            marker=dict(
                size=max(20, int(d["priority"] * 60)),
                color=color,
                opacity=0.85,
                line=dict(color=CARD_BG, width=2),
            ),
            text=[d["title"][:20] + "…" if len(d["title"]) > 20 else d["title"]],
            textposition="top center",
            textfont=dict(color=TEXT_MAIN, size=10),
            name=d["title"][:30],
            hovertemplate=(
                f"<b>{d['title']}</b><br>"
                f"Impact: {'Low' if d['impact']==1 else 'Medium' if d['impact']==2 else 'High'}<br>"
                f"Effort: {'Low' if d['effort']==1 else 'Medium' if d['effort']==2 else 'High'}<br>"
                f"Priority: {round(d['priority']*100)}%<br>"
                f"Owner: {d['owner']}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        **_base(spec.title, spec.subtitle, height=420),
        xaxis=dict(
            **_ax(),
            range=[0.3, 3.7],
            tickvals=[1, 2, 3],
            ticktext=["Low effort", "Medium effort", "High effort"],
            title="Effort to execute",
        ),
        yaxis=dict(
            **_ax(),
            range=[0.3, 3.7],
            tickvals=[1, 2, 3],
            ticktext=["Low impact", "Medium impact", "High impact"],
            title="Business impact",
        ),
    )
    return fig


# ── Chart 6: Correlation annotation ──────────────────────────────────────────

def _render_correlation(spec: ChartSpec) -> go.Figure:
    r     = spec.meta.get("pearson_r", 0)
    p_val = spec.meta.get("p_value", 1)
    cols  = spec.columns

    # Visual representation of r value on a -1 to +1 scale
    abs_r = abs(r)
    color = "#34d399" if r > 0 else "#ef4444"

    fig = go.Figure()

    # r-value gauge bar
    fig.add_trace(go.Bar(
        x=[r],
        y=["Correlation"],
        orientation="h",
        marker=dict(
            color=color,
            opacity=0.8,
            line=dict(color=CARD_BG, width=1),
        ),
        width=0.4,
        text=[f"r = {r:.3f}"],
        textposition="outside",
        textfont=dict(color=TEXT_MAIN, size=13),
        hovertemplate=f"Pearson r = {r:.3f}<br>p-value = {p_val:.4f}<extra></extra>",
        name="Correlation strength",
    ))

    # Zero line
    fig.add_vline(x=0, line=dict(color=TEXT_MUTED, width=1))

    # Strength zones
    for x0, x1, label, fc in [
        (-1.0, -0.7, "Strong negative", "rgba(239,68,68,0.06)"),
        (-0.7, -0.3, "Moderate negative", "rgba(239,68,68,0.03)"),
        (-0.3,  0.3, "Weak / no correlation", "rgba(148,163,184,0.03)"),
        ( 0.3,  0.7, "Moderate positive", "rgba(52,211,153,0.03)"),
        ( 0.7,  1.0, "Strong positive", "rgba(52,211,153,0.06)"),
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=fc, line_width=0)
        fig.add_annotation(
            x=(x0+x1)/2, y=0.85,
            text=label, showarrow=False,
            font=dict(color=TEXT_MUTED, size=9),
            yref="paper",
        )

    col_str = " ↔ ".join(c.replace("_"," ") for c in cols)
    fig.update_layout(
        **_base(spec.title, f"{col_str} · p-value: {p_val:.4f}", height=220),
        xaxis=dict(**_ax(), range=[-1.1, 1.4], title="Correlation coefficient (r)"),
        yaxis=dict(**_ax()),
        bargap=0.5,
    )
    return fig