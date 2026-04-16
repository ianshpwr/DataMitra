import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.data_agent.agent import DataProcessingAgent

agent = DataProcessingAgent()

def test_live_batch():
    """
    Requires:
      1. Redis running:  redis-server
      2. Simulator running:  python data/live/stream_simulator.py
    """
    print("Waiting 3 seconds for stream events to accumulate...")
    time.sleep(3)

    result = agent.run_live_batch(batch_size=50)

    if result is None:
        print("No live events yet — make sure stream_simulator.py is running")
        return

    clean_df, ctx = result
    assert ctx.source_type == "live_stream"
    assert ctx.domain      == "ecommerce"
    assert ctx.row_count    > 0

    print(f"\nPassed live test — {ctx.row_count} events processed")
    print(f"Processing time: {ctx.processing_ms}ms")
    print(clean_df.head(3))

if __name__ == "__main__":
    test_live_batch()