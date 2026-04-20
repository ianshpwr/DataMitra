import re
import polars as pl
from agents.insight_agent.models import Insight, InsightBundle, InsightType, InsightSeverity
from agents.data_agent.models    import DataContext
from .models import InsightVerdict, CriticReport

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_CONFIDENCE_TO_PASS    = 0.50
FLAG_CONFIDENCE_THRESHOLD = 0.65
MIN_EVIDENCE_COUNT        = 1
MAX_TITLE_WORDS           = 20
MIN_EXPLANATION_WORDS     = 15

# ── Flag config ───────────────────────────────────────────────────────────────
CRITICAL_FLAGS   = ["column_not_in_data", "evidence_value_implausible"]
FLAG_PENALTY     = 0.15
MAX_FLAG_PENALTY = 0.45

# ── Insight types whose evidence values ARE the distribution (never flag) ─────
STATISTICAL_SUMMARY_TYPES = {
    InsightType.summary,
    InsightType.distribution,
    InsightType.correlation,
}

# ── Metric names that represent aggregates, never individual outliers ─────────
SUMMARY_METRICS = {
    "mean", "median", "std", "q1", "q3", "count",
    "null_pct", "null_count", "n_unique", "top_pct",
    "bottom_pct", "imbalance_ratio", "outlier_pct",
    "slope", "r_squared", "p_value", "trend_pct_per_period",
    "pearson_r", "anomaly_pct", "column_mean", "column_std",
    "skew", "kurtosis",
}

VAGUE_PHRASES = [
    "may indicate", "could be", "might suggest", "possibly",
    "it appears", "seems like", "unclear", "unknown",
]


class CriticAgent:
    """
    Agent 3 — Critic Agent.

    Validates each Insight against the raw DataFrame + DataContext.
    No LLM — pure rule-based + statistical grounding.

    Scoring rubric (max = 1.0 before penalties):
      +0.35  data_grounded   — every affected column exists in the DataFrame
      +0.25  title quality   — concise, specific, no vague language or markdown
      +0.20  explanation ok  — enough words, not too vague
      +0.20  evidence ok     — at least one StatEvidence item attached
      -0.15  per flag        — each flag reduces score (max penalty -0.45)

    Critical flags (column_not_in_data, evidence_value_implausible)
    block PASS regardless of score.

    Evidence plausibility is only checked for anomaly and trend insights —
    summary/distribution/correlation insights report aggregate statistics
    by design and must never be flagged as implausible.
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

        # ── Apply confidence scores, filter strictly by verdict.passed ────────
        verdict_map = {v.insight_id: v for v in verdicts}
        passed_insights = []
        for ins in bundle.insights:
            v = verdict_map.get(ins.id)
            ins.confidence = v.confidence_score if v else 0.0
            if v and v.passed:
                passed_insights.append(ins)

        passed   = [v for v in verdicts if v.passed]
        rejected = [v for v in verdicts if not v.passed]
        flagged  = [v for v in verdicts if len(v.flags) > 0]

        avg_quality = (
            sum(v.confidence_score for v in passed) / len(passed)
            if passed else 0.0
        )

        # ── Human review trigger ──────────────────────────────────────────────
        needs_review = (
            any(
                any(cf in f for cf in CRITICAL_FLAGS)
                for v in verdicts
                for f in v.flags
            )
            or len(flagged) > 2
        )

        # ── Retry trigger ─────────────────────────────────────────────────────
        retry = (
            len(rejected) > len(passed)
            or (avg_quality < 0.55 and len(bundle.insights) > 2)
        )

        report = CriticReport(
            passed_count=len(passed),
            rejected_count=len(rejected),
            flagged_count=len(flagged),
            overall_quality=round(avg_quality, 3),
            needs_human_review=needs_review,
            verdicts=verdicts,
            retry_recommended=retry,
            retry_reason=(
                f"Only {len(passed)}/{len(verdicts)} insights passed "
                f"with avg quality {avg_quality:.2f}"
            ) if retry else None,
        )

        validated_bundle = bundle.model_copy(
            update={"insights": passed_insights}
        )

        return validated_bundle, report

    # ── Per-insight evaluation ────────────────────────────────────────────────

    def _evaluate(
        self,
        insight: Insight,
        df: pl.DataFrame,
        ctx: DataContext,
    ) -> InsightVerdict:

        score = 0.0
        flags = []

        # ── Check 1: Data grounding (+0.35) ──────────────────────────────────
        missing_cols = [
            c for c in insight.affected_columns
            if c not in df.columns
        ]
        if not missing_cols:
            score += 0.35
        else:
            flags.append(f"column_not_in_data: {missing_cols}")

        # ── Check 2: Evidence plausibility ───────────────────────────────────
        # Only run for anomaly + trend insights.
        # Summary, distribution, and correlation insights report aggregate
        # statistics (means, percentiles, correlations) as evidence — these
        # are correct by construction and must never be z-scored against
        # the same column they describe.
        should_check_plausibility = insight.type not in STATISTICAL_SUMMARY_TYPES

        if should_check_plausibility:
            for ev in insight.evidence:
                # Skip any metric that is itself a distributional aggregate
                if ev.metric in SUMMARY_METRICS:
                    continue

                if ev.column and ev.column in df.columns:
                    col_series = df[ev.column].drop_nulls()
                    if str(col_series.dtype) in (
                        "Float64", "Float32", "Int64", "Int32"
                    ):
                        try:
                            # Support scalar or list evidence values
                            if isinstance(ev.value, list):
                                check_vals = [float(x) for x in ev.value]
                            else:
                                check_vals = [
                                    float(
                                        str(ev.value)
                                        .replace("$", "")
                                        .replace(",", "")
                                    )
                                ]

                            col_mean = float(col_series.mean())
                            col_std  = float(col_series.std()) or 1.0
                            q1       = float(col_series.quantile(0.25))
                            q3       = float(col_series.quantile(0.75))
                            iqr      = q3 - q1
                            lower    = q1 - 3.0 * iqr
                            upper    = q3 + 3.0 * iqr

                            for val in check_vals:
                                # Skip if value is within 5% of column mean —
                                # catches renamed summary stats
                                if col_mean != 0:
                                    if abs(val - col_mean) / abs(col_mean) < 0.05:
                                        continue

                                z           = abs(val - col_mean) / col_std
                                outside_iqr = val < lower or val > upper

                                # Both z > 3 AND outside IQR = genuine outlier
                                if z > 3 and outside_iqr:
                                    flags.append(
                                        f"evidence_value_implausible: "
                                        f"{ev.metric}={val}"
                                    )

                        except (ValueError, TypeError):
                            flags.append(
                                f"invalid_numeric_evidence: {ev.metric}"
                            )

        # ── Check 3: Title quality (+0.25) ───────────────────────────────────
        title_words = len(insight.title.split())
        has_vague   = any(p in insight.title.lower() for p in VAGUE_PHRASES)
        has_bold    = "**" in insight.title
        has_number  = bool(re.search(r"\d", insight.title))

        if title_words <= MAX_TITLE_WORDS and not has_vague and not has_bold:
            score += 0.20
            if has_number:
                score += 0.05
        else:
            if has_vague:
                flags.append("vague_title_language")
            if has_bold:
                flags.append("markdown_in_title")
            if title_words > MAX_TITLE_WORDS:
                flags.append(f"title_too_long: {title_words} words")

        # ── Check 4: Explanation quality (+0.20) ─────────────────────────────
        explanation_words = len(insight.explanation.split())
        vague_count       = sum(
            1 for p in VAGUE_PHRASES
            if p in insight.explanation.lower()
        )

        if explanation_words >= MIN_EXPLANATION_WORDS and vague_count <= 1:
            score += 0.20
        else:
            if explanation_words < MIN_EXPLANATION_WORDS:
                flags.append(
                    f"explanation_too_short: {explanation_words} words"
                )
            if vague_count > 1:
                flags.append(f"too_vague: {vague_count} vague phrases")

        # ── Check 5: Evidence present (+0.20) ────────────────────────────────
        if len(insight.evidence) >= MIN_EVIDENCE_COUNT:
            score += 0.20
        else:
            flags.append("no_evidence_attached")

        # ── Apply flag penalty ────────────────────────────────────────────────
        penalty = min(len(flags) * FLAG_PENALTY, MAX_FLAG_PENALTY)
        score  -= penalty
        score   = round(max(0.0, min(score, 1.0)), 3)

        # ── Critical flag check — blocks PASS regardless of score ─────────────
        has_critical_flag = any(
            any(cf in f for cf in CRITICAL_FLAGS)
            for f in flags
        )

        passed = score >= MIN_CONFIDENCE_TO_PASS and not has_critical_flag

        return InsightVerdict(
            insight_id=insight.id,
            passed=passed,
            confidence_score=score,
            rejection_reason=(
                "Critical flag: "
                + next(
                    f for f in flags
                    if any(cf in f for cf in CRITICAL_FLAGS)
                )
                if has_critical_flag else
                f"Score too low: {score}"
                if not passed else None
            ),
            flags=flags,
        )