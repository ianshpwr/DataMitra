import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from agents.data_agent.agent   import DataProcessingAgent
from agents.insight_agent.agent import InsightAgent

def test_insight_agent_static():
    print("\n── Stage 1: Data Agent ──────────────────────────────")
    data_agent = DataProcessingAgent()
    clean_df, ctx = data_agent.run_static("data/static/orders.csv")
    print(f"Data loaded: {ctx.row_count} rows, domain={ctx.domain}")

    print("\n── Stage 2: Insight Agent ───────────────────────────")
    insight_agent = InsightAgent()
    bundle = insight_agent.run(clean_df, ctx)

    print(f"\n{'='*55}")
    print(f"EXECUTIVE SUMMARY")
    print(f"{'='*55}")
    print(bundle.executive_summary)

    print(f"\n{'='*55}")
    print(f"INSIGHTS ({len(bundle.insights)} found)")
    print(f"{'='*55}")
    for ins in bundle.insights:
        icon = {"critical": "!!!", "warning": " ! ", "info": "   "}[ins.severity]
        print(f"\n[{icon}] [{ins.type.upper()}] {ins.title}")
        print(f"     {ins.explanation}")
        print(f"     Evidence: {ins.evidence[0].metric}={ins.evidence[0].value}"
              if ins.evidence else "")

    print(f"\n{'='*55}")
    print(f"Stats: {len(bundle.insights)} insights | "
          f"{bundle.analysis_ms}ms | {bundle.token_count} tokens")

    assert len(bundle.insights) > 0
    assert bundle.executive_summary
    print("\nAll insight agent tests passed.")

if __name__ == "__main__":
    test_insight_agent_static()