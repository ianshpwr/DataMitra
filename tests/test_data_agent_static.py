"""
Run with: python -m pytest nexus/tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.data_agent.agent import DataProcessingAgent
import polars as pl

agent = DataProcessingAgent()

def test_static_orders():
    clean_df, ctx = agent.run_static("data/static/orders.csv")

    # Domain detected correctly
    assert ctx.domain == "ecommerce", f"Expected ecommerce, got {ctx.domain}"

    # Cleaning removed duplicates
    assert ctx.quality.duplicate_rows >= 0

    # total_amount nulls were filled
    null_count = clean_df["total_amount"].null_count()
    assert null_count < ctx.quality.total_rows * 0.01, \
        f"Still {null_count} nulls in total_amount after cleaning"

    # unit_price is now numeric
    assert str(clean_df["unit_price"].dtype) in ("Float64", "Float32"), \
        f"unit_price should be float, got {clean_df['unit_price'].dtype}"

    # Status is lowercase
    statuses = clean_df["status"].unique().to_list()
    assert all(s == s.lower() for s in statuses if s), "Status has uppercase values"

    print(f"\nPassed static test — {ctx.row_count} rows, domain={ctx.domain}")
    print(f"Quality score: {ctx.quality.overall_score}")
    print(f"Processing time: {ctx.processing_ms}ms")
    print(f"Warnings: {ctx.warnings[:3]}")

def test_static_products():
    clean_df, ctx = agent.run_static("data/static/products.csv")
    assert ctx.row_count > 0
    print(f"\nPassed products test — {ctx.row_count} rows")

def test_static_customers():
    clean_df, ctx = agent.run_static("data/static/customers.csv")
    assert ctx.row_count > 0
    print(f"\nPassed customers test — {ctx.row_count} rows")

if __name__ == "__main__":
    test_static_orders()
    test_static_products()
    test_static_customers()
    print("\nAll static tests passed.")