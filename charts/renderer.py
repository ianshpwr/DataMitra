"""
Renders ChartPlan → Plotly Figure using REAL DataFrame data.
No simulated distributions. No fake stats.
"""
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from agents.chart_agent.models import ChartPlan

CARD_BG    = "#1e2130"
DARK_BG    = "#0f1117"
BORDER     = "#2d3150"
GRID       = "#252b3b"
TEXT_MAIN  = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
PALETTE    = ["#60a5fa","#34d399","#f59e0b","#f87171",
              "#a78bfa","#fb923c","#38bdf8","#4ade80"]


def _base(title="", subtitle="", height=360):
    anns = []
    if subtitle:
        anns.append(dict(
            text=subtitle, xref="paper", yref="paper",
            x=0, y=1.07, showarrow=False,
            font=dict(color=TEXT_MUTED, size=11), align="left",
        ))
    return dict(
        title=dict(text=title, font=dict(color=TEXT_MAIN, size=15), x=0),
        annotations=anns,
        height=height,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_MUTED, size=12),
        margin=dict(l=20, r=20, t=72, b=20),
        legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_MUTED, size=11)),
    )

def _ax(**kwargs):
    base = dict(gridcolor=GRID, linecolor=BORDER,
                tickfont=dict(color=TEXT_MUTED, size=11),
                title_font=dict(color=TEXT_MUTED))
    base.update(kwargs)
    return base

def render_from_data(plan, df: pl.DataFrame) -> go.Figure | None:
    # Accept both ChartPlan object and raw dict
    if isinstance(plan, dict):
        from agents.chart_agent.models import ChartPlan
        plan = ChartPlan(**plan)

    try:
        ct = plan.chart_type
        if ct == "bar":
            return _bar(plan, df)
        if ct == "histogram":
            return _histogram(plan, df)
        if ct == "line":
            return _line(plan, df)
        if ct == "scatter":
            return _scatter(plan, df)
        if ct == "pie":
            return _pie(plan, df)
        if ct == "heatmap":
            return _heatmap(plan, df)
    except Exception as e:
        chart_id = plan.chart_type if hasattr(plan, "chart_type") else str(plan)
        insight_id = plan.insight_id if hasattr(plan, "insight_id") else "unknown"
        print(f"[Renderer] Failed {chart_id} for {insight_id}: {e}")
    return None
# ── Bar ───────────────────────────────────────────────────────────────────────

def _bar(plan: ChartPlan, df: pl.DataFrame) -> go.Figure | None:
    col = plan.x_column
    if not col or col not in df.columns:
        # fallback: find first categorical col in df
        for c in df.columns:
            if str(df[c].dtype) == "String" and df[c].n_unique() <= 30:
                col = c
                break
    if not col:
        return None

    agg   = plan.aggregation or "count"
    y_col = plan.y_column

    if agg == "count" or not y_col or y_col not in df.columns:
        counts = (
            df[col].drop_nulls()
              .value_counts()
              .sort("count", descending=True)
              .head(12)
        )
        labels = counts[col].to_list()
        values = counts["count"].to_list()
        y_title = "Count"
    else:
        agg_df = (
            df.group_by(col)
              .agg(pl.col(y_col).mean().alias("value"))
              .sort("value", descending=True)
              .head(12)
        )
        labels = agg_df[col].to_list()
        values = [round(v, 2) for v in agg_df["value"].to_list()]
        y_title = f"Mean {y_col.replace('_',' ')}"

    total  = sum(values) or 1
    pcts   = [f"{round(v/total*100,1)}%" for v in values]
    colors = PALETTE[:len(labels)]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color=DARK_BG, width=1)),
        text=pcts, textposition="outside",
        textfont=dict(color=TEXT_MUTED, size=10),
        hovertemplate="<b>%{x}</b><br>" + y_title + ": %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **_base(plan.title, plan.subtitle, height=360),
        xaxis=_ax(title=col.replace("_"," ").title()),
        yaxis=_ax(title=y_title),
        bargap=0.25,
    )
    return fig


# ── Histogram ─────────────────────────────────────────────────────────────────

def _histogram(plan: ChartPlan, df: pl.DataFrame) -> go.Figure | None:
    col = plan.x_column
    if not col or col not in df.columns:
        return None

    series = df[col].drop_nulls()
    if str(series.dtype) not in ("Float64","Float32","Int64","Int32","Int16","Int8"):
        return None

    vals   = series.to_list()
    mean   = float(series.mean() or 0)
    std    = float(series.std()  or 1)
    q3     = float(series.quantile(0.75) or mean)
    iqr    = float((series.quantile(0.75) or mean) - (series.quantile(0.25) or 0))
    thresh = q3 + 1.5 * iqr

    # Split normal vs outlier
    normal   = [v for v in vals if v <= thresh]
    outliers = [v for v in vals if v >  thresh]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=normal,
        nbinsx=40,
        marker=dict(color="rgba(96,165,250,0.7)",
                    line=dict(color=DARK_BG, width=0.5)),
        name="Normal values",
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))

    if outliers:
        fig.add_trace(go.Histogram(
            x=outliers,
            nbinsx=10,
            marker=dict(color="rgba(239,68,68,0.7)",
                        line=dict(color=DARK_BG, width=0.5)),
            name=f"Outliers ({len(outliers)})",
            hovertemplate="Outlier: %{x:,.2f}<extra></extra>",
        ))

    fig.add_vline(
        x=mean,
        line=dict(color="#34d399", width=1.5, dash="dash"),
        annotation=dict(text=f"Mean: {mean:,.2f}",
                        font=dict(color="#34d399", size=10),
                        bgcolor=CARD_BG),
    )
    fig.add_vline(
        x=thresh,
        line=dict(color="#f59e0b", width=1, dash="dot"),
        annotation=dict(text="Outlier threshold",
                        font=dict(color="#f59e0b", size=10),
                        bgcolor=CARD_BG),
    )

    fig.update_layout(
        **_base(plan.title, plan.subtitle, height=360),
        barmode="overlay",
        xaxis=_ax(title=col.replace("_"," ").title()),
        yaxis=_ax(title="Count"),
    )
    return fig


# ── Line ──────────────────────────────────────────────────────────────────────

def _line(plan: ChartPlan, df: pl.DataFrame) -> go.Figure | None:
    date_col = plan.x_column
    val_col  = plan.y_column

    # Find date column automatically if not set
    if not date_col or date_col not in df.columns:
        for c in df.columns:
            if any(k in c.lower() for k in ["date","time","month","year"]):
                date_col = c
                break
    if not date_col:
        return None

    # Find value column
    if not val_col or val_col not in df.columns:
        for c in df.columns:
            if str(df[c].dtype) in ("Float64","Float32","Int64","Int32"):
                val_col = c
                break
    if not val_col:
        return None

    try:
        ts = (
            df.with_columns(
                pl.col(date_col).str.slice(0,7).alias("_month")
            )
            .group_by("_month")
            .agg(pl.col(val_col).sum().alias("value"))
            .sort("_month")
            .drop_nulls("_month")
        )
        if len(ts) < 2:
            return None

        months = ts["_month"].to_list()
        values = ts["value"].to_list()

        import numpy as np
        x_num   = list(range(len(values)))
        slope   = float(np.polyfit(x_num, values, 1)[0])
        trend_y = [float(np.polyfit(x_num, values, 1)[0]*i
                         + np.polyfit(x_num, values, 1)[1])
                   for i in x_num]
        trend_color = "#34d399" if slope >= 0 else "#ef4444"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=values,
            fill="tozeroy", fillcolor="rgba(96,165,250,0.1)",
            line=dict(color="#60a5fa", width=2),
            mode="lines+markers",
            marker=dict(size=5, color="#60a5fa"),
            name=val_col.replace("_"," ").title(),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=months, y=trend_y,
            line=dict(color=trend_color, width=2, dash="dash"),
            mode="lines",
            name="Trend",
        ))
        fig.update_layout(
            **_base(plan.title, plan.subtitle, height=360),
            xaxis=_ax(title="Month"),
            yaxis=_ax(title=val_col.replace("_"," ").title()),
        )
        return fig
    except Exception as e:
        print(f"[Renderer] Line chart failed: {e}")
        return None


# ── Scatter ───────────────────────────────────────────────────────────────────

def _scatter(plan: ChartPlan, df: pl.DataFrame) -> go.Figure | None:
    x_col = plan.x_column
    y_col = plan.y_column

    if not x_col or not y_col:
        return None
    if x_col not in df.columns or y_col not in df.columns:
        return None

    sample = df.select([x_col, y_col]).drop_nulls().sample(
        n=min(500, len(df)), seed=42
    )
    x_vals = sample[x_col].to_list()
    y_vals = sample[y_col].to_list()

    import numpy as np
    x_arr   = np.array(x_vals, dtype=float)
    y_arr   = np.array(y_vals, dtype=float)
    slope   = np.polyfit(x_arr, y_arr, 1)
    trend_x = [float(min(x_arr)), float(max(x_arr))]
    trend_y = [float(np.polyval(slope, v)) for v in trend_x]

    color_vals = None
    color_name = None
    if plan.color_col and plan.color_col in df.columns:
        color_vals = sample[plan.color_col].cast(str).to_list()
        color_name = plan.color_col

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(size=5, color="#60a5fa", opacity=0.6,
                    line=dict(color=DARK_BG, width=0.5)),
        name="Data points",
        hovertemplate=(
            f"{x_col}: %{{x:,.2f}}<br>"
            f"{y_col}: %{{y:,.2f}}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=trend_x, y=trend_y,
        mode="lines",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        name="Trend line",
    ))
    fig.update_layout(
        **_base(plan.title, plan.subtitle, height=380),
        xaxis=_ax(title=x_col.replace("_"," ").title()),
        yaxis=_ax(title=y_col.replace("_"," ").title()),
    )
    return fig


# ── Pie ───────────────────────────────────────────────────────────────────────

def _pie(plan: ChartPlan, df: pl.DataFrame) -> go.Figure | None:
    col = plan.x_column
    if not col or col not in df.columns:
        return None

    vc = (
        df[col].drop_nulls()
          .value_counts()
          .sort("count", descending=True)
          .head(6)
    )
    labels = vc[col].to_list()
    values = vc["count"].to_list()

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker=dict(colors=PALETTE[:len(labels)],
                    line=dict(color=DARK_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color=TEXT_MAIN, size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_base(plan.title, plan.subtitle, height=320))
    return fig


# ── Heatmap ───────────────────────────────────────────────────────────────────

def _heatmap(plan: ChartPlan, df: pl.DataFrame) -> go.Figure | None:
    num_cols = [
        c for c in df.columns
        if str(df[c].dtype) in ("Float64","Float32","Int64","Int32")
    ][:8]

    if len(num_cols) < 2:
        return None

    import numpy as np
    data = df.select(num_cols).drop_nulls().to_numpy()
    corr = np.corrcoef(data.T)

    labels = [c.replace("_"," ") for c in num_cols]

    fig = go.Figure(go.Heatmap(
        z=corr,
        x=labels, y=labels,
        colorscale=[
            [0.0, "#ef4444"],
            [0.5, CARD_BG],
            [1.0, "#34d399"],
        ],
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr],
        texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT_MAIN),
        hovertemplate="<b>%{x} × %{y}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color=TEXT_MUTED),
            title=dict(text="r", font=dict(color=TEXT_MUTED)),
        ),
    ))
    fig.update_layout(
        **_base(plan.title, plan.subtitle, height=max(320, len(num_cols)*50)),
        xaxis=_ax(),
        yaxis=_ax(),
    )
    return fig