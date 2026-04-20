import os
import json
from groq import Groq
from dotenv import load_dotenv
import numpy as np  # ✅ added

from .models import Insight, InsightType, InsightSeverity, StatEvidence
from ..data_agent.models import DataContext
from .stats_engine import StatResult

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FAST_MODEL = "llama-3.1-8b-instant"
DEEP_MODEL = "llama-3.3-70b-versatile"

_total_tokens = 0


def _sanitize_value(v):
    """
    Ensures value passed to StatEvidence is valid (float/int/str).
    Handles list, dict, and other complex types safely.
    """

    # ✅ CASE 1: list → mean
    if isinstance(v, list):
        try:
            return float(sum(v) / len(v)) if len(v) > 0 else 0.0
        except Exception:
            return str(v)

    # ✅ CASE 2: dict → pick best representative value
    if isinstance(v, dict):
        # priority order (very important for analytics clarity)
        for key in ["mean", "avg", "median", "value", "count", "sum", "max", "min"]:
            if key in v:
                try:
                    return float(v[key])
                except Exception:
                    return str(v[key])

        # fallback
        return str(v)

    # ✅ CASE 3: already valid
    if isinstance(v, (int, float, str)):
        return v

    # ✅ fallback for anything weird
    return str(v)

def explain_stat_result(
    stat: StatResult,
    ctx: DataContext,
    insight_id: str,
    use_fast_model: bool = False,
) -> tuple[Insight, int]:

    model = FAST_MODEL if (use_fast_model or stat.severity == "info") else DEEP_MODEL
    prompt = _build_prompt(stat, ctx)

    response = client.chat.completions.create(
        model=model,
        max_tokens=400,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior data analyst explaining statistical findings "
                    "to business executives. Be concise, specific, and avoid jargon."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens
    _add_tokens(tokens_used)

    title, explanation = _parse_response(raw_text, stat)

    # ✅ FIX APPLIED HERE
    evidence = [
        StatEvidence(
            metric=k,
            value=_sanitize_value(v),  # 🔥 critical fix
            column=stat.columns[0] if stat.columns else None,
        )
        for k, v in list(stat.metrics.items())[:4]
    ]

    return Insight(
        id=insight_id,
        type=_map_type(stat.stat_type),
        severity=InsightSeverity(stat.severity),
        title=title,
        explanation=explanation,
        evidence=evidence,
        affected_columns=stat.columns,
        confidence=0.0,
        actionable=stat.actionable,
        raw_stats=stat.metrics,
    ), tokens_used


def generate_executive_summary(
    insights: list[Insight],
    ctx: DataContext,
) -> tuple[str, int]:

    insight_titles = "\n".join(
        f"- [{i.severity.upper()}] {i.title}" for i in insights[:8]
    )

    prompt = f"""Write a 3-sentence executive summary for a {ctx.domain} dataset \
with {ctx.row_count:,} rows.

Key findings:
{insight_titles}

Structure:
- Sentence 1: Overall state of the data
- Sentence 2: Most critical issue/opportunity
- Sentence 3: Most important next action

Output only the 3 sentences."""

    response = client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=200,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a concise executive business analyst.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    tokens = response.usage.total_tokens
    _add_tokens(tokens)
    return response.choices[0].message.content.strip(), tokens


def reset_token_counter():
    global _total_tokens
    _total_tokens = 0


def get_total_tokens() -> int:
    return _total_tokens


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_tokens(n: int):
    global _total_tokens
    _total_tokens += n


def _build_prompt(stat: StatResult, ctx: DataContext) -> str:
    safe_metrics = {
        k: v for k, v in list(stat.metrics.items())[:6]
        if not isinstance(v, list) or len(v) <= 5
    }
    col_names = [c.name for c in ctx.columns]

    return f"""Dataset: {ctx.domain} business data | {ctx.row_count:,} rows
Columns available: {col_names}
Finding type: {stat.stat_type}
Affected columns: {stat.columns}
Statistics:
{json.dumps(safe_metrics, indent=2)}

Write exactly 2 lines:
Line 1: Title (max 12 words)
Line 2: Explanation (2–3 sentences)

No jargon."""

def _parse_response(text: str, stat: StatResult) -> tuple[str, str]:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    # Strip markdown bold/italic the LLM sometimes adds
    def clean(s: str) -> str:
        import re
        return re.sub(r'\*+', '', s).strip()

    if len(lines) >= 2:
        return clean(lines[0]), clean(" ".join(lines[1:]))
    elif len(lines) == 1:
        return clean(lines[0]), clean(lines[0])
    else:
        return (
            f"{stat.stat_type.replace('_', ' ').title()} in {stat.columns}",
            "A notable statistical pattern was detected in this data.",
        )

def _map_type(stat_type: str) -> InsightType:
    return {
        "summary": InsightType.summary,
        "time_trend": InsightType.trend,
        "anomaly": InsightType.anomaly,
        "correlation": InsightType.correlation,
        "distribution": InsightType.distribution,
        "categorical_distribution": InsightType.distribution,
        "high_nulls": InsightType.summary,
    }.get(stat_type, InsightType.summary)