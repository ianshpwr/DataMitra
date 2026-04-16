import polars as pl
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

CATEGORIES = {
    "Electronics": ["Laptop", "Phone", "Tablet", "Headphones", "Camera"],
    "Clothing":    ["T-Shirt", "Jeans", "Jacket", "Shoes", "Dress"],
    "Home":        ["Sofa", "Lamp", "Rug", "Pillow", "Frame"],
    "Sports":      ["Yoga Mat", "Dumbbells", "Running Shoes", "Bottle"],
    "Books":       ["Fiction", "Non-Fiction", "Textbook", "Comic"],
}
REGIONS    = ["North", "South", "East", "West", "Central"]
STATUSES   = ["completed", "COMPLETED", "Completed",   # intentional case mess
               "pending", "Pending",
               "cancelled", "Cancelled"]
PAYMENTS   = ["credit_card", "debit_card", "paypal", "upi", "cod"]

# ── Products ──────────────────────────────────────────────────────────────────
def generate_products(n=300):
    rows = []
    for i in range(1, n + 1):
        cat     = random.choice(list(CATEGORIES))
        product = random.choice(CATEGORIES[cat])
        cost    = round(random.uniform(5, 500), 2)
        sell    = round(cost * random.uniform(1.2, 2.5), 2)
        rows.append({
            "product_id":      f"P{i:04d}",
            "product_name":    f"{product} {fake.word().capitalize()}",
            "category":        cat,
            "sub_category":    product,
            "cost_price":      cost,
            "selling_price":   sell,
            "stock_quantity":  random.randint(0, 500),
            "supplier_id":     f"SUP{random.randint(1, 30):03d}",
        })
    return pl.DataFrame(rows)

# ── Customers ─────────────────────────────────────────────────────────────────
def generate_customers(n=1500):
    rows = []
    for i in range(1, n + 1):
        orders = random.randint(1, 50)
        rows.append({
            "customer_id":    f"C{i:05d}",
            "signup_date":    fake.date_between(start_date="-3y", end_date="today").isoformat(),
            "age":            random.randint(18, 70) if random.random() > 0.05 else None,  # 5% missing
            "city":           fake.city(),
            "country":        random.choice(["India", "India", "India", "US", "UK", "UAE"]),
            "email":          fake.email(),
            "total_orders":   orders,
            "lifetime_value": round(orders * random.uniform(30, 300), 2),
            "churn_risk":     random.choice(["low", "medium", "high"]),
        })
    return pl.DataFrame(rows)

# ── Orders ────────────────────────────────────────────────────────────────────
def generate_orders(products_df, customers_df, n=4000):
    product_ids  = products_df["product_id"].to_list()
    prices       = dict(zip(
        products_df["product_id"].to_list(),
        products_df["selling_price"].to_list()
    ))
    customer_ids = customers_df["customer_id"].to_list()
    rows         = []

    for i in range(1, n + 1):
        pid      = random.choice(product_ids)
        cid      = random.choice(customer_ids)
        qty      = random.randint(1, 10)
        price    = prices[pid]
        total    = round(qty * price, 2)

        # Intentional date format mess
        order_date = fake.date_between(start_date="-1y", end_date="today")
        if random.random() < 0.3:
            date_str = order_date.strftime("%d/%m/%Y")  # wrong format
        else:
            date_str = order_date.isoformat()           # correct ISO

        # Intentional price format mess (string with $)
        if random.random() < 0.2:
            price_str = f"${price}"
        else:
            price_str = str(price)

        # Intentional missing total (8%)
        total_val = None if random.random() < 0.08 else total

        rows.append({
            "order_id":       f"ORD{i:06d}",
            "customer_id":    cid,
            "product_id":     pid,
            "order_date":     date_str,
            "quantity":       qty,
            "unit_price":     price_str,    # messy
            "total_amount":   total_val,    # 8% None
            "status":         random.choice(STATUSES),  # case mess
            "payment_method": random.choice(PAYMENTS),
            "region":         random.choice(REGIONS),
        })

    # Add 3% duplicate rows
    dupes = random.sample(rows, k=int(n * 0.03))
    rows.extend(dupes)
    random.shuffle(rows)

    return pl.DataFrame(rows)

if __name__ == "__main__":
    out = os.path.dirname(__file__)
    print("Generating products...")
    products  = generate_products(300)
    products.write_csv(f"{out}/products.csv")
    print(f"  products.csv  → {len(products)} rows")

    print("Generating customers...")
    customers = generate_customers(1500)
    customers.write_csv(f"{out}/customers.csv")
    print(f"  customers.csv → {len(customers)} rows")

    print("Generating orders...")
    orders    = generate_orders(products, customers, 4000)
    orders.write_csv(f"{out}/orders.csv")
    print(f"  orders.csv    → {len(orders)} rows (includes duplicates)")

    print("\nDone. Files written to nexus/data/static/")