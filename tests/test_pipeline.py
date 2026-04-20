import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from agents.pipeline import run_static_pipeline

def print_divider(title: str):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")

def test_full_pipeline():
    print_divider("DataMitra — Full Pipeline Test")

    result = run_static_pipeline("data/static/orders.csv")

    # ── Critic report ────────────────────────────────────────────────────────
    report = result["critic_report"]
    print_divider("Critic Report")
    print(f"  Passed:          {report.passed_count}")
    print(f"  Rejected:        {report.rejected_count}")
    print(f"  Flagged:         {report.flagged_count}")
    print(f"  Quality score:   {report.overall_quality:.2f}")
    print(f"  Needs review:    {report.needs_human_review}")
    print(f"  Retry was used:  {result['retry_count'] > 0}")

    print("\n  Per-insight verdicts:")
    for v in report.verdicts:
        status = "PASS" if v.passed else "FAIL"
        print(f"  [{status}] {v.insight_id}  conf={v.confidence_score:.2f}  "
              f"flags={v.flags or 'none'}")

    # ── Final validated insights ─────────────────────────────────────────────
    bundle = result["final_bundle"]
    print_divider(f"Validated Insights ({len(bundle.insights)} passed)")
    print(f"\n  {bundle.executive_summary}\n")

    for ins in bundle.insights:
        bar = "█" * int(ins.confidence * 10) + "░" * (10 - int(ins.confidence * 10))
        print(f"  [{ins.severity.upper():<8}] {ins.title}")
        print(f"  Confidence: [{bar}] {ins.confidence:.2f}")
        print(f"  {ins.explanation[:120]}...")
        print()

    # ── Assertions ───────────────────────────────────────────────────────────
    assert result["completed"],        "Pipeline did not complete"
    assert bundle is not None,         "No final bundle produced"
    assert len(bundle.insights) > 0,   "No insights survived critic"
    assert report.overall_quality > 0, "Quality score is zero"
    assert result.get("error") is None, f"Pipeline error: {result.get('error')}"

    print_divider("All pipeline tests passed")

if __name__ == "__main__":
    test_full_pipeline()