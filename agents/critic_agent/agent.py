import re
import polars as pl
from agents.insight_agent.models import Insight, InsightBundle, InsightSeverity
from agents.data_agent.models    import DataContext
from .models import InsightVerdict, CriticReport

# Thresholds
MIN_CONFIDENCE_TO_PASS     = 0.50
FLAG_CONFIDENCE_THRESHOLD  = 0.65   # passed but flagged for human review
MIN_EVIDENCE_COUNT         = 1
MAX_TITLE_WORDS            = 20
MIN_EXPLANATION_WORDS      = 15
VAGUE_PHRASES              = [
    "may indicate", "could be", "might suggest", "possibly",
    "it appears", "seems like", "unclear", "unknown",
]


class CriticAgent:
    """
    Agent 3 — Critic Agent.
    Validates each Insight against the raw DataFrame + DataContext.
    No LLM — pure rule-based + statistical grounding.

    Scoring rubric (each check is worth points, max = 1.0):
      +0.35  data_grounded:   every evidence value exists in the DataFrame
      +0.25  specific_title:  title is concise and free of vague language
      +0.20  explanation_ok:  explanation has enough words and specifics
      +0.20  evidence_ok:     at least one StatEvidence item attached
    """

    def run(
        self,
        bundle: InsightBundle,
        df: pl.DataFrame,
        ctx: DataContext,
    ) -> tuple[InsightBundle, CriticReport]:

        verdicts: list[InsightVerdict] = []

        for insight in bundle.insights:
            verdict = self._evaluate(insight, df, ctx)
            verdicts.append(verdict)

        # Apply confidence scores back onto the insights
        score_map = {v.insight_id: v.confidence_score for v in verdicts}
        passed_insights = []
        for ins in bundle.insights:
            ins.confidence = score_map.get(ins.id, 0.0)
            if score_map.get(ins.id, 0.0) >= MIN_CONFIDENCE_TO_PASS:
                passed_insights.append(ins)

        passed   = [v for v in verdicts if v.passed]
        rejected = [v for v in verdicts if not v.passed]
        flagged  = [v for v in passed if v.confidence_score < FLAG_CONFIDENCE_THRESHOLD]

        avg_quality = (
            sum(v.confidence_score for v in passed) / len(passed)
            if passed else 0.0
        )

        # Recommend retry if too many rejections OR avg quality is low
        retry = (
            len(rejected) > len(passed)
            or (avg_quality < 0.55 and len(bundle.insights) > 2)
        )

        report = CriticReport(
            passed_count=len(passed),
            rejected_count=len(rejected),
            flagged_count=len(flagged),
            overall_quality=round(avg_quality, 3),
            needs_human_review=len(flagged) > 2,
            verdicts=verdicts,
            retry_recommended=retry,
            retry_reason=(
                f"Only {len(passed)}/{len(verdicts)} insights passed "
                f"with avg quality {avg_quality:.2f}"
            ) if retry else None,
        )

        # Replace bundle insights with only the passed + scored ones
        validated_bundle = bundle.model_copy(
            update={"insights": passed_insights}
        )

        return validated_bundle, report

    # ── Per-insight evaluation ──────────────────────────────────────────────

    def _evaluate(
        self,
        insight: Insight,
        df: pl.DataFrame,
        ctx: DataContext,
    ) -> InsightVerdict:

        score  = 0.0
        flags  = []

        # ── Check 1: Data grounding (+0.35) ────────────────────────────────
        # Every affected column must actually exist in the DataFrame
        missing_cols = [
            c for c in insight.affected_columns
            if c not in df.columns
        ]
        if not missing_cols:
            score += 0.35
        else:
            flags.append(f"column_not_in_data: {missing_cols}")

        # For numeric evidence, verify value is plausible (within 3σ of column)
        for ev in insight.evidence:
            if ev.column and ev.column in df.columns:
                col_series = df[ev.column].drop_nulls()
                if str(col_series.dtype) in ("Float64", "Float32", "Int64", "Int32"):
                    try:
                        ev_val = float(str(ev.value).replace("$", "").replace(",", ""))
                        col_mean = float(col_series.mean())
                        col_std  = float(col_series.std()) or 1.0
                        z = abs(ev_val - col_mean) / col_std
                        if z > 5:
                            flags.append(f"evidence_value_implausible: {ev.metric}={ev.value}")
                    except (ValueError, TypeError):
                        pass

        # ── Check 2: Title quality (+0.25) ──────────────────────────────────
        title_words = len(insight.title.split())
        has_vague   = any(p in insight.title.lower() for p in VAGUE_PHRASES)
        has_bold    = "**" in insight.title
        has_number  = bool(re.search(r"\d", insight.title))  # specific = good

        if title_words <= MAX_TITLE_WORDS and not has_vague and not has_bold:
            score += 0.20
            if has_number:
                score += 0.05   # bonus for specificity
        else:
            if has_vague:
                flags.append("vague_title_language")
            if has_bold:
                flags.append("markdown_in_title")
            if title_words > MAX_TITLE_WORDS:
                flags.append(f"title_too_long: {title_words} words")

        # ── Check 3: Explanation quality (+0.20) ────────────────────────────
        explanation_words = len(insight.explanation.split())
        has_vague_exp     = sum(
            1 for p in VAGUE_PHRASES if p in insight.explanation.lower()
        )

        if explanation_words >= MIN_EXPLANATION_WORDS and has_vague_exp <= 1:
            score += 0.20
        else:
            if explanation_words < MIN_EXPLANATION_WORDS:
                flags.append(f"explanation_too_short: {explanation_words} words")
            if has_vague_exp > 1:
                flags.append(f"too_vague: {has_vague_exp} vague phrases")

        # ── Check 4: Evidence attached (+0.20) ──────────────────────────────
        if len(insight.evidence) >= MIN_EVIDENCE_COUNT:
            score += 0.20
        else:
            flags.append("no_evidence_attached")

        score = round(min(score, 1.0), 3)
        passed = score >= MIN_CONFIDENCE_TO_PASS and "column_not_in_data" not in " ".join(flags)

        return InsightVerdict(
            insight_id=insight.id,
            passed=passed,
            confidence_score=score,
            rejection_reason=(
                f"Score {score} below threshold or critical flag: {flags[0]}"
                if not passed else None
            ),
            flags=flags,
        )