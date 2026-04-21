import time
import polars as pl
from agents.insight_agent.models  import InsightBundle
from agents.data_agent.models     import DataContext
from .models          import Decision, DecisionPack
from .rules           import match_rule
from .llm_recommender import (
    generate_decision,
    generate_decision_summary,
    reset_token_counter,
    get_total_tokens,
)

# Only generate decisions for actionable insights
# or high-severity ones regardless of actionable flag
def _should_decide(insight) -> bool:
    return (
        insight.actionable
        or insight.severity in ("critical", "warning")
    )


class DecisionAgent:
    """
    Agent 4 — Decision Agent.
    Input:  validated InsightBundle + DataContext
    Output: DecisionPack (ranked action recommendations)
    """

    def run(
        self,
        bundle: InsightBundle,
        ctx:    DataContext,
    ) -> DecisionPack:

        start = time.time()
        reset_token_counter()

        # Filter to insights worth deciding on
        target_insights = [
            ins for ins in bundle.insights
            if _should_decide(ins)
        ]

        print(f"  [DecisionAgent] Generating decisions for "
              f"{len(target_insights)}/{len(bundle.insights)} insights")

        decisions: list[Decision] = []

        for i, insight in enumerate(target_insights):
            rule = match_rule(
                insight_type=insight.type,
                severity=insight.severity,
                title=insight.title,
                domain=bundle.domain,
            )

            print(f"  [DecisionAgent] Decision {i+1}/{len(target_insights)}: "
                  f"{insight.type} ({insight.severity}) → {rule.action_type}")

            decision, _ = generate_decision(
                insight=insight,
                rule=rule,
                domain=bundle.domain,
                decision_id=f"DEC-{i+1:03d}",
            )
            decisions.append(decision)

        # Sort by priority score descending
        decisions.sort(key=lambda d: d.priority_score, reverse=True)

        # Quick wins = high/medium impact + low effort
        quick_wins = [
            d for d in decisions
            if d.impact_level in ("high", "medium")
            and d.effort_level == "low"
        ]

        # Executive action summary
        summary = ""
        if decisions:
            print("  [DecisionAgent] Generating action summary...")
            summary, _ = generate_decision_summary(decisions, bundle.domain)

        elapsed_ms = int((time.time() - start) * 1000)

        return DecisionPack(
            domain=bundle.domain,
            total_insights=len(bundle.insights),
            decisions=decisions,
            top_priority=decisions[0] if decisions else None,
            quick_wins=quick_wins,
            summary=summary,
            generation_ms=elapsed_ms,
            token_count=get_total_tokens(),
        )