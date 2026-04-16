import polars as pl
import time
from datetime import datetime
from typing import Literal

from .loader   import StaticLoader, LiveStreamLoader
from .profiler import profile_dataframe
from .validator import compute_quality_report
from .cleaner  import clean_dataframe
from .models   import DataContext

# Domain fingerprinting: column name patterns → domain label
DOMAIN_SIGNATURES = {
    "ecommerce": {"order_id", "product_id", "customer_id",
                  "unit_price", "total_amount", "quantity"},
    "finance":   {"transaction_id", "account_id", "debit",
                  "credit", "balance", "instrument"},
    "saas":      {"user_id", "subscription_id", "mrr",
                  "churn", "plan", "trial"},
    "hr":        {"employee_id", "department", "salary",
                  "hire_date", "performance"},
}


def detect_domain(df: pl.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    scores = {
        domain: len(sig & cols) / len(sig)
        for domain, sig in DOMAIN_SIGNATURES.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0.3 else "unknown"


class DataProcessingAgent:
    """
    Agent 1 — Data Processing Agent.
    Accepts static files or live stream batches.
    Returns a cleaned DataFrame + DataContext metadata object.
    """

    def __init__(self):
        self.static_loader = StaticLoader()
        self.live_loader   = LiveStreamLoader()

    # ── Static path ──────────────────────────────────────────────────────────
    def run_static(
        self,
        file_path: str,
        use_sample: bool = False,
        sample_size: int = 10_000,
    ) -> tuple[pl.DataFrame, DataContext]:

        start = time.time()

        # 1. Load
        raw_df = (
            self.static_loader.load_sample(file_path, sample_size)
            if use_sample
            else self.static_loader.load(file_path)
        )

        # 2. Profile raw (before cleaning — catches original issues)
        raw_profile = profile_dataframe(raw_df)
        domain      = detect_domain(raw_df)
        quality     = compute_quality_report(raw_df)

        # 3. Clean
        clean_df = clean_dataframe(raw_df)

        # 4. Collect warnings
        warnings = []
        if not quality.passed:
            warnings.append(
                f"Quality score {quality.overall_score:.2f} — review columns: "
                + ", ".join(quality.columns_failing)
            )
        for col_profile in raw_profile:
            for issue in col_profile.issues:
                warnings.append(f"Column '{col_profile.name}': {issue}")

        elapsed_ms = int((time.time() - start) * 1000)

        ctx = DataContext(
            source_type="static_csv",
            domain=domain,
            row_count=len(clean_df),
            column_count=len(clean_df.columns),
            columns=raw_profile,
            quality=quality,
            processing_ms=elapsed_ms,
            warnings=warnings,
        )

        return clean_df, ctx

    # ── Live stream path ──────────────────────────────────────────────────────
    def run_live_batch(
        self,
        batch_size: int = 100,
    ) -> tuple[pl.DataFrame, DataContext] | None:

        start  = time.time()
        raw_df = self.live_loader.consume_batch(batch_size)

        if raw_df.is_empty():
            return None  # no data right now — caller decides what to do

        profile  = profile_dataframe(raw_df)
        domain   = detect_domain(raw_df)
        quality  = compute_quality_report(raw_df)
        clean_df = clean_dataframe(raw_df)

        elapsed_ms = int((time.time() - start) * 1000)

        ctx = DataContext(
            source_type="live_stream",
            domain=domain,
            row_count=len(clean_df),
            column_count=len(clean_df.columns),
            columns=profile,
            quality=quality,
            processing_ms=elapsed_ms,
        )

        return clean_df, ctx