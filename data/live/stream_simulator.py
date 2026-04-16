"""
Simulates a live order event stream.
Uses Redis LIST as a lightweight message queue.
Swap redis_client.rpush() → kafka_producer.send() when ready for production.
"""
import redis
import json
import time
import random
from datetime import datetime
from faker import Faker

fake   = Faker()
client = redis.Redis(host="localhost", port=6379, decode_responses=True)

TOPIC      = "nexus:live:orders"
PRODUCTS   = [f"P{i:04d}" for i in range(1, 301)]
CUSTOMERS  = [f"C{i:05d}" for i in range(1, 1501)]
STATUSES   = ["completed", "pending", "cancelled"]
PAYMENTS   = ["credit_card", "debit_card", "paypal", "upi", "cod"]
REGIONS    = ["North", "South", "East", "West", "Central"]

order_counter = 100_000

def generate_live_event():
    global order_counter
    order_counter += 1
    qty   = random.randint(1, 5)
    price = round(random.uniform(10, 999), 2)
    return {
        "order_id":       f"LIVE{order_counter:08d}",
        "customer_id":    random.choice(CUSTOMERS),
        "product_id":     random.choice(PRODUCTS),
        "order_date":     datetime.utcnow().isoformat(),
        "quantity":       qty,
        "unit_price":     price,
        "total_amount":   round(qty * price, 2),
        "status":         random.choice(STATUSES),
        "payment_method": random.choice(PAYMENTS),
        "region":         random.choice(REGIONS),
        "source":         "live_stream",
        "ingested_at":    datetime.utcnow().isoformat(),
    }

def run_stream(events_per_second: float = 1.0, max_events: int = None):
    delay   = 1.0 / events_per_second
    emitted = 0
    print(f"Stream started → Redis LIST '{TOPIC}' at {events_per_second} events/sec")
    print("Press Ctrl+C to stop\n")
    try:
        while True:
            event = generate_live_event()
            client.rpush(TOPIC, json.dumps(event))
            emitted += 1
            print(f"  [{emitted}] {event['order_id']} | {event['status']} | ${event['total_amount']:.2f}")
            if max_events and emitted >= max_events:
                break
            time.sleep(delay + random.uniform(0, delay * 0.5))  # slight jitter
    except KeyboardInterrupt:
        print(f"\nStream stopped. {emitted} events emitted.")

if __name__ == "__main__":
    run_stream(events_per_second=0.8)  # ~1 event per 1.25 seconds