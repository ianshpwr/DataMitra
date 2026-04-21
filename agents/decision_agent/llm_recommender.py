import os
import json
from groq import Groq
from dotenv import load_dotenv
from .models import Decision, ActionType, ImpactLevel
from .rules  import RuleMatch
from agents.insight_agent.models import Insight

load_dotenv()
client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
FAST_MODEL = "llama-3.1-8b-instant"
DEEP_MODEL = "llama-3.3-70b-versatile"

_total_tokens = 0


def generate_decision(
    insight:    Insight,
    rule:       RuleMatch,
    domain:     str,
    decision_id: str,
) -> tuple[Decision, int]:
    global _total_tokens

    # Use deep model only for critical/actionable insights
    model = DEEP_MODEL if (
        insight.severity in ("critical", "warning") and insight.actionable
    ) else FAST_MODEL

    prompt = f"""You are a senior business consultant giving a specific, concrete action recommendation.

Domain: {domain}
Insight type: {insight.type}
Severity: {insight.severity}
Insight title: {insight.title}
Insight explanation: {insight.explanation}
Evidence: {json.dumps([{"metric": e.metric, "value": e.value} for e in insight.evidence[:3]])}
Suggested action type: {rule.action_type}
Suggested owner: {rule.owner}
KPI to track: {rule.kpi}

Respond with ONLY a valid JSON object. No explanation, no markdown, no preamble.
Use exactly these keys:
{{
  "title": "Action title (max 10 words, start with a verb)",
  "what": "Exactly what to do — specific steps (2-3 sentences)",
  "why": "Why this action addresses the insight (1-2 sentences)",
  "expected_impact": "What measurable outcome to expect (1 sentence with a number if possible)",
  "risk_if_ignored": "What happens if this is not done (1 sentence)"
}}"""

    response = client.chat.completions.create(
        model=model,
        max_tokens=500,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise business consultant. "
                    "You always respond with valid JSON only. No markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw      = response.choices[0].message.content.strip()
    tokens   = response.usage.total_tokens
    _total_tokens += tokens

    parsed = _safe_parse(raw)

    # Compute priority score: base from severity + rule boost + confidence
    severity_base = {"critical": 0.7, "warning": 0.5, "info": 0.2}
    priority = min(
        severity_base.get(insight.severity, 0.2)
        + rule.priority_boost
        + (insight.confidence * 0.1),
        1.0,
    )

    return Decision(
        id=decision_id,
        insight_id=insight.id,
        action_type=rule.action_type,
        title=parsed.get("title", f"Address {insight.title[:40]}"),
        what=parsed.get("what", "Investigate this finding and determine root cause."),
        why=parsed.get("why", insight.explanation[:200]),
        expected_impact=parsed.get("expected_impact", "Improved data quality and decision accuracy."),
        impact_level=rule.impact_level,
        effort_level=rule.effort_level,
        priority_score=round(priority, 3),
        risk_if_ignored=parsed.get("risk_if_ignored", "Issue may compound over time."),
        owner=rule.owner,
        kpi=rule.kpi,
    ), tokens


def generate_decision_summary(
    decisions: list[Decision],
    domain:    str,
) -> tuple[str, int]:
    global _total_tokens

    top = sorted(decisions, key=lambda d: d.priority_score, reverse=True)[:3]
    top_titles = "\n".join(f"- [{d.impact_level.upper()}] {d.title}" for d in top)

    prompt = f"""Write a 2-sentence action summary for a {domain} business.

Top priority actions:
{top_titles}

Sentence 1: What is the single most important thing to do right now and why.
Sentence 2: What overall outcome these actions will achieve if executed.

Plain text only. No bullet points, no markdown."""

    response = client.chat.completions.create(
        model=FAST_MODEL,
        max_tokens=150,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a concise executive advisor."},
            {"role": "user",   "content": prompt},
        ],
    )
    tokens = response.usage.total_tokens
    _total_tokens += tokens
    return response.choices[0].message.content.strip(), tokens


def reset_token_counter():
    global _total_tokens
    _total_tokens = 0

def get_total_tokens() -> int:
    return _total_tokens


def _safe_parse(raw: str) -> dict:
    """Parse JSON from LLM output — handles markdown fences gracefully."""
    import re
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$",          "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}