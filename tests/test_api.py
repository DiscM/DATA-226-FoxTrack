import csv
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from foxtrack.api.app import create_app

N_FEATURES = 6


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("service")
    (root / "models").mkdir(); (root / "demo").mkdir()
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(N_FEATURES, 2)))
    mean = rng.normal(size=N_FEATURES)
    thresholds = [
        {"target_fpr": 1e-4, "alert_rule": "threshold", "smooth_window": 1, "k": 1, "n": 1, "threshold": 1.0},
        {"target_fpr": 1e-4, "alert_rule": "3 of 5", "smooth_window": 1, "k": 3, "n": 5, "threshold": 1.0},
        {"target_fpr": 1e-4, "alert_rule": "mean of 100", "smooth_window": 100, "k": 1, "n": 1, "threshold": 0.5},
    ]
    model = {"name": "mirai_botnet", "kind": "pca_reconstruction_error", "n_features": N_FEATURES, "k_components": 2,
             "mean": mean.tolist(), "std": [1.0] * N_FEATURES, "components": q.T.tolist(), "thresholds": thresholds,
             "trained_on": {"first_attack_idx": 2000}}
    (root / "models" / "mirai_botnet.json").write_text(json.dumps(model))

    features = (mean + rng.normal(size=(3000, N_FEATURES)) * 0.5).astype(np.float32)
    features[2000:] += 3.0
    labels = (np.arange(3000) >= 2000).astype(np.int8)
    np.savez_compressed(root / "demo" / "mirai_botnet.npz", features=features, labels=labels,
                        first_idx=np.int64(100_000), attack_start=np.int64(2000))
    with open(root / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["attack", "target_fpr", "alert_rule", "delay_packets", "delay_credible"])
        writer.writeheader()
        writer.writerow({"attack": "mirai_botnet", "target_fpr": 0.0001, "alert_rule": "mean of 100", "delay_packets": 42.0, "delay_credible": True})
        writer.writerow({"attack": "mirai_botnet", "target_fpr": 0.001, "alert_rule": "threshold", "delay_packets": "", "delay_credible": False})
    app = create_app(root / "models", root / "demo", root / "results.csv")
    test_client = TestClient(app)
    test_client.features = features
    return test_client


def test_health_and_models(client):
    assert client.get("/api/health").json() == {"status": "ok", "models": 1, "sessions": 0}
    models = client.get("/api/models").json()
    assert models[0]["name"] == "mirai_botnet" and models[0]["title"] == "Mirai"
    assert models[0]["has_demo"] and models[0]["n_features"] == N_FEATURES
    assert {"alert_rule": "mean of 100", "target_fpr": 1e-4} in models[0]["settings"]


def test_dashboard_page(client):
    page = client.get("/")
    assert page.status_code == 200 and "FoxTrack" in page.text


def test_predict(client):
    rows = client.features[1990:2010].tolist()
    out = client.post("/api/predict", json={"model": "mirai_botnet", "rows": rows, "alert_rule": "threshold"}).json()
    assert len(out["scores"]) == 20 and len(out["alerts"]) == 20
    assert not any(out["alerts"][:10]) and all(out["alerts"][10:])
    assert out["alert_events"] == 1


def test_predict_errors(client):
    row = client.features[0].tolist()
    assert client.post("/api/predict", json={"model": "none", "rows": [row]}).status_code == 404
    assert client.post("/api/predict", json={"model": "mirai_botnet", "rows": [[1.0, 2.0]]}).status_code == 422
    assert client.post("/api/predict", json={"model": "mirai_botnet", "rows": []}).status_code == 422
    assert client.post("/api/predict", json={"model": "mirai_botnet", "rows": [row], "alert_rule": "bad"}).status_code == 422


def test_session_keeps_history_between_requests(client):
    whole = client.post("/api/predict", json={"model": "mirai_botnet", "rows": client.features.tolist()}).json()
    session = client.post("/api/sessions", json={"model": "mirai_botnet"})
    assert session.status_code == 201
    sid = session.json()["session_id"]
    alerts = []
    for i in range(0, 3000, 250):
        part = client.post(f"/api/sessions/{sid}/packets", json={"rows": client.features[i:i + 250].tolist()}).json()
        alerts += part["alerts"]
    assert alerts == whole["alerts"]
    assert part["packets_seen"] == 3000
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.post(f"/api/sessions/{sid}/packets", json={"rows": [client.features[0].tolist()]}).status_code == 404


def test_demo_replay(client):
    out = client.get("/api/demo/mirai_botnet").json()
    assert out["attack_start"] == 2000 and len(out["values"]) == 3000 and out["first_idx"] == 100_000
    assert out["summary"]["detected"] and out["summary"]["delay_packets"] < 100
    assert out["summary"]["false_alarm_events"] == 0 and out["summary"]["packet_recall"] > 0.9
    assert client.get("/api/demo/none").status_code == 404
    assert client.get("/api/demo/mirai_botnet", params={"target_fpr": 0.5}).status_code == 422


def test_results_table(client):
    rows = client.get("/api/results").json()
    assert len(rows) == 1 and rows[0]["delay_packets"] == 42.0 and rows[0]["delay_credible"] is True
    other = client.get("/api/results", params={"alert_rule": "threshold", "target_fpr": 0.001}).json()
    assert other[0]["delay_packets"] is None
