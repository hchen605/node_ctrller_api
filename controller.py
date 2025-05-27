# controller.py
import numpy as np
import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional

# Map model version → worker base URL
WORKERS = {
    1: "http://0.0.0.0:5001",
    2: "http://0.0.0.0:5002",
}

app = FastAPI(title="MNIST-Controller")

class MNISTRequest(BaseModel):
    data: list  # flattened 784‐length list or nested list of shape [1,1,28,28]

def _fetch_worker_metrics(url: str) -> dict:
    """
    Hits GET {url}/metrics and returns its JSON payload.
    Expected payload: {"version": int, "val_accuracy": float}
    """
    resp = requests.get(f"{url}/metrics", timeout=1)
    resp.raise_for_status()
    return resp.json()

def _choose_best_version() -> int:
    best_ver, best_acc = None, -1.0
    for ver, base in WORKERS.items():
        try:
            m = _fetch_worker_metrics(base)
            if m["val_accuracy"] is not None and m["val_accuracy"] > best_acc:
                best_acc, best_ver = m["val_accuracy"], ver
        except Exception:
            continue
    if best_ver is None:
        raise RuntimeError("No worker metrics available")
    return best_ver

@app.post("/predict")
def predict(req: MNISTRequest,
            worker: Optional[int] = Query(None, description="force a specific model version")):
    # 1) Decide which version to call
    if worker is None:
        worker = _choose_best_version()
    if worker not in WORKERS:
        return {"error": f"Unknown version {version}"}

    base_url = WORKERS[worker]
    
    # 2) Forward the payload to the worker’s /invocations
    invoc = {"inputs": np.array(req.data, dtype=np.float32).reshape(1,1,28,28).tolist()}
    r = requests.post(f"{base_url}/invocations", json=invoc, timeout=2)
    r.raise_for_status()
    logits = np.array(r.json())

    # 3) Fetch val_accuracy from that worker again for the response
    metrics = _fetch_worker_metrics(base_url)

    return {
        "WORKER":      worker,
        "Sample Prediction":   int(logits.argmax()),
        "Model Name:": metrics.get("name", None),
        "Model Version:": metrics.get("version", None),
        "Model val_accuracy": float(metrics.get("val_accuracy", None))
    }

# --- optional: standalone demo ----------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("controller:app", host="0.0.0.0", port=9000, log_level="info")
