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
    assert client.get("/api/health").json() == {"status": "ok", "models": 1, "sessions": 0, "model_source": "file"}
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


# ---- Models from an MLflow record ----

@pytest.fixture(scope="module")
def mlflow_record(tmp_path_factory):
    """A small MLflow record with the layout of a Kaggle run: mlflow.db and mlruns/."""
    mlflow = pytest.importorskip("mlflow")
    root = tmp_path_factory.mktemp("mlflow_record")
    work = tmp_path_factory.mktemp("training_machine")   # the place of the files at training time
    rng = np.random.default_rng(3)

    def model_dict(name, threshold):
        q, _ = np.linalg.qr(rng.normal(size=(N_FEATURES, 2)))
        return {"name": name, "kind": "pca_reconstruction_error", "n_features": N_FEATURES, "k_components": 2,
                "mean": [0.0] * N_FEATURES, "std": [1.0] * N_FEATURES, "components": q.T.tolist(),
                "thresholds": [{"target_fpr": 1e-4, "alert_rule": "mean of 100", "smooth_window": 100, "k": 1, "n": 1,
                                "threshold": threshold}], "trained_on": {}}

    mlflow.set_tracking_uri(f"sqlite:///{work}/mlflow.db")
    mlflow.create_experiment("baseline_pca_onset", artifact_location=str(work / "mlruns" / "1"))
    mlflow.set_experiment("baseline_pca_onset")
    # Two runs for os_scan: the service must select the newer one (threshold 0.9).
    for name, threshold in (("os_scan", 0.1), ("mirai_botnet", 0.5), ("os_scan", 0.9)):
        path = work / f"{name}.json"
        path.write_text(json.dumps(model_dict(name, threshold)))
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"attack": name, "k_components": 2})
            mlflow.log_artifact(str(path), artifact_path="model")
    with mlflow.start_run(run_name="no_model"):
        mlflow.log_param("note", "a run with no attack and no model")
    # Move the record to a different folder, as after a download from Kaggle.
    import shutil
    shutil.copy(work / "mlflow.db", root / "mlflow.db")
    shutil.copytree(work / "mlruns", root / "mlruns")
    shutil.rmtree(work)
    return root


def test_models_from_mlflow_record(mlflow_record, tmp_path):
    from foxtrack.serving.mlflow_store import load_models
    before = (mlflow_record / "mlflow.db").read_bytes()
    models = load_models(mlflow_record)
    assert list(models) == ["mirai_botnet", "os_scan"]
    assert models["os_scan"].threshold("mean of 100", 1e-4) == 0.9   # the newer run
    assert models["os_scan"].source["type"] == "mlflow" and len(models["os_scan"].source["run_id"]) == 32
    assert models["os_scan"].source["params"]["attack"] == "os_scan"
    assert (mlflow_record / "mlflow.db").read_bytes() == before      # the record is not changed

    app = create_app(tmp_path / "no_models", tmp_path / "no_demo", tmp_path / "none.csv", mlflow_dir=mlflow_record)
    client = TestClient(app)
    assert client.get("/api/health").json()["model_source"] == "mlflow"
    listed = {m["name"]: m for m in client.get("/api/models").json()}
    assert listed["mirai_botnet"]["source"]["experiment"] == "baseline_pca_onset"
    row = [0.0] * N_FEATURES
    assert client.post("/api/predict", json={"model": "os_scan", "rows": [row]}).status_code == 200


def test_mlflow_record_errors(mlflow_record, tmp_path):
    from foxtrack.serving.mlflow_store import load_models
    with pytest.raises(FileNotFoundError, match="No MLflow database"):
        load_models(tmp_path)
    with pytest.raises(ValueError, match="no experiment"):
        load_models(mlflow_record, experiment="other")
    import shutil
    broken = tmp_path / "broken"; broken.mkdir()
    shutil.copy(mlflow_record / "mlflow.db", broken / "mlflow.db")
    with pytest.raises(FileNotFoundError, match="has no model file"):
        load_models(broken)
