import polars as pl
import time
import os
from dotenv import load_dotenv

from .stats_engine  import run_full_analysis
from .llm_explainer import (
    explain_stat_result,
    generate_executive_summary,
    reset_token_counter,
    get_total_tokens,
    DEEP_MODEL,
)
from .models        import InsightBundle
from ..data_agent.models import DataContext

load_dotenv()

# Only explain the top N findings to control LLM cost
MAX_INSIGHTS_TO_EXPLAIN = 8


class InsightAgent:
    """
    Agent 2 — Insight Agent.
    Input:  cleaned DataFrame + DataContext from DataProcessingAgent
    Output: InsightBundle (list of natural-language insights + executive summary)
    """

    def run(
        self,
        df: pl.DataFrame,
        ctx: DataContext,
        max_insights: int = MAX_INSIGHTS_TO_EXPLAIN,
    ) -> InsightBundle:

        start = time.time()
        reset_token_counter()

        # ── Stage 1: Pure statistical analysis (fast, no LLM) ──
        print("  [InsightAgent] Running statistical analysis...")
        stat_results = run_full_analysis(df, ctx.domain)
        print(f"  [InsightAgent] Found {len(stat_results)} statistical findings")

        # ── Stage 2: Select top findings to explain ──
        # Always explain critical + warning; fill remainder with info
        critical_warn = [r for r in stat_results
                         if r.severity in ("critical", "warning")]
        info_only     = [r for r in stat_results if r.severity == "info"]
        to_explain    = (critical_warn + info_only)[:max_insights]

        # ── Stage 3: LLM explanation for each (fast model for info, deep for critical) ──
        insights = []
        for i, stat in enumerate(to_explain):
            print(f"  [InsightAgent] Explaining {i+1}/{len(to_explain)}: "
                  f"{stat.stat_type} ({stat.severity})")
            use_fast = stat.severity == "info"
            insight, _ = explain_stat_result(
                stat, ctx,
                insight_id=f"INS-{i+1:03d}",
                use_fast_model=use_fast,
            )
            insights.append(insight)

        # ── Stage 4: Executive summary ──
        print("  [InsightAgent] Generating executive summary...")
        exec_summary, _ = generate_executive_summary(insights, ctx)

        elapsed_ms = int((time.time() - start) * 1000)
        total_tokens = get_total_tokens()

        bundle = InsightBundle(
            domain=ctx.domain,
            source_type=ctx.source_type,
            total_rows=ctx.row_count,
            insights=insights,
            executive_summary=exec_summary,
            analysis_ms=elapsed_ms,
            llm_model_used=DEEP_MODEL,
            token_count=total_tokens,
        )

        return bundle