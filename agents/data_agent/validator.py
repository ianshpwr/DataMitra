import polars as pl
from .models import QualityReport

def compute_quality_report(
    df: pl.DataFrame,
    null_threshold: float   = 0.30,
    duplicate_threshold_pct: float = 0.05,
) -> QualityReport:
    """
    Lightweight quality check without full Great Expectations overhead.
    Great Expectations integration shown in Phase 2.
    """
    total_rows     = len(df)
    duplicate_rows = total_rows - len(df.unique())
    duplicate_pct  = duplicate_rows / total_rows if total_rows > 0 else 0

    # Per-column null check
    null_row_vals = [
        df[c].null_count() / total_rows
        for c in df.columns
    ]
    avg_null_pct      = sum(null_row_vals) / len(null_row_vals) if null_row_vals else 0
    columns_failing   = [
        df.columns[i]
        for i, pct in enumerate(null_row_vals)
        if pct > null_threshold
    ]

    # Scoring: start at 1.0, penalise each issue
    score = 1.0
    score -= min(avg_null_pct * 2, 0.4)          # nulls penalty (max -0.4)
    score -= min(duplicate_pct * 3, 0.3)         # duplicates penalty (max -0.3)
    score -= len(columns_failing) * 0.05         # failing columns
    score  = max(0.0, round(score, 3))

    return QualityReport(
        total_rows=total_rows,
        duplicate_rows=duplicate_rows,
        null_row_pct=round(avg_null_pct, 4),
        columns_failing=columns_failing,
        overall_score=score,
        passed=score >= 0.6 and duplicate_pct < duplicate_threshold_pct,
    )