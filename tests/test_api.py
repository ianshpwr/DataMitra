import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print(f"Health: {data}")

def test_analyse_upload():
    file_path = "data/static/orders.csv"
    with open(file_path, "rb") as f:
        r = client.post(
            "/api/v1/analyse/upload",
            files={"file": ("orders.csv", f, "text/csv")},
            params={"use_sample": True},   # sample = faster test
        )

    assert r.status_code == 200, f"Failed: {r.text}"
    data = r.json()

    print(f"\nDomain:          {data['domain']}")
    print(f"Total rows:      {data['total_rows']}")
    print(f"Insights:        {len(data['insights'])}")
    print(f"Quality score:   {data['quality']['overall_score']}")
    print(f"Processing ms:   {data['processing_ms']}")
    print(f"\nExecutive summary:\n{data['executive_summary']}")
    print(f"\nFirst insight: {data['insights'][0]['title']}")

    assert data["success"]
    assert data["domain"] == "ecommerce"
    assert len(data["insights"]) > 0
    assert data["quality"]["overall_score"] > 0.5

def test_analyse_path():
    r = client.post(
        "/api/v1/analyse/path",
        params={"file_path": "data/static/orders.csv", "use_sample": True},
    )
    assert r.status_code == 200
    assert r.json()["success"]
    print(f"\nPath endpoint: {r.json()['total_rows']} rows analysed")

def test_bad_extension():
    r = client.post(
        "/api/v1/analyse/upload",
        files={"file": ("data.xlsx", b"fake", "application/octet-stream")},
    )
    assert r.status_code == 400
    print(f"\nBad extension correctly rejected: {r.json()['detail']}")

if __name__ == "__main__":
    test_health()
    test_analyse_upload()
    test_analyse_path()
    test_bad_extension()
    print("\nAll API tests passed.")