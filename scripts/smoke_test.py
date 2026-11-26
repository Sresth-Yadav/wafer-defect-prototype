import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from dashboard.main import app

img_path = ROOT / "Dataset" / "images" / "images" / "deburring" / "debur_img_002.jpg"

with TestClient(app) as c:
    r_home = c.get("/")
    print("GET /:", r_home.status_code)
    r_health = c.get("/api/health")
    print("GET /api/health:", r_health.status_code, r_health.json())
    r_metrics = c.get("/api/metrics")
    print("GET /api/metrics acc:", r_metrics.status_code, r_metrics.json()["model"]["overall_accuracy"])
    with img_path.open("rb") as f:
        r_run = c.post("/run", files={"file": ("debur_img_002.jpg", f, "image/jpeg")})
    print("POST /run:", r_run.status_code, r_run.json()["output"]["inference"], "severity:", r_run.json()["output"]["severity"])
