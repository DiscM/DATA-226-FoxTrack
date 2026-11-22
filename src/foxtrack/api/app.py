"""FastAPI service for the FoxTrack detector.

Start it from the repository root:

    uvicorn foxtrack.api.app:app --app-dir src

Then open http://127.0.0.1:8000 for the dashboard, or /docs for the API.

The service loads one trained model for each recording from ``models/pca_baseline``.
Scoring uses NumPy only. Spark is necessary only for training.
"""

from __future__ import annotations

import csv
import os
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from foxtrack.metrics import onset_evaluation
from foxtrack.serving import OnlineDetector, PcaModel

REPO = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
MAX_ROWS = 20_000        # rows in one request
MAX_SESSIONS = 200       # the oldest session is removed when this number is reached

TITLES = {
    "arp_mitm": "ARP MitM", "active_wiretap": "Active Wiretap", "fuzzing": "Fuzzing",
    "mirai_botnet": "Mirai", "os_scan": "OS Scan", "ssdp_flood": "SSDP Flood",
    "ssl_renegotiation": "SSL Renegotiation", "syn_dos": "SYN DoS", "video_injection": "Video Injection",
}


class PredictRequest(BaseModel):
    model: str = Field(description="Name of the model, for example 'mirai_botnet'")
    rows: list[list[float]] = Field(description="Packets in sequence. Each packet has 115 feature values.")
    alert_rule: str = "mean of 100"
    target_fpr: float = 1e-4


class SessionRequest(BaseModel):
    model: str
    alert_rule: str = "mean of 100"
    target_fpr: float = 1e-4


class PacketsRequest(BaseModel):
    rows: list[list[float]]


def _plain(value):
    """Convert NumPy values to plain Python values for JSON."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def _number(text: str):
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text) if text != "" else None
    except ValueError:
        return text


def create_app(models_dir: Path | None = None, demo_dir: Path | None = None, results_csv: Path | None = None) -> FastAPI:
    models_dir = Path(models_dir or os.environ.get("FOXTRACK_MODELS_DIR", REPO / "models" / "pca_baseline"))
    demo_dir = Path(demo_dir or os.environ.get("FOXTRACK_DEMO_DIR", REPO / "demo_data"))
    results_csv = Path(results_csv or os.environ.get("FOXTRACK_RESULTS_CSV", REPO / "results" / "baseline_pca.csv"))

    models = {p.stem: PcaModel.load(p) for p in sorted(models_dir.glob("*.json"))}
    sessions: OrderedDict[str, OnlineDetector] = OrderedDict()

    app = FastAPI(
        title="FoxTrack",
        version="1.0",
        description="Early attack detection on network traffic. PCA baseline trained on benign traffic of each recording.",
    )

    def get_model(name: str) -> PcaModel:
        if name not in models:
            raise HTTPException(404, f"Unknown model '{name}'. Available: {sorted(models)}")
        return models[name]

    def get_detector(model: PcaModel, alert_rule: str, target_fpr: float) -> OnlineDetector:
        try:
            return OnlineDetector(model, alert_rule, target_fpr)
        except KeyError as error:
            raise HTTPException(422, f"{error.args[0]}. Available settings: {model.settings()}")

    def run(detector: OnlineDetector, rows) -> dict:
        if not rows:
            raise HTTPException(422, "The request has no rows")
        if len(rows) > MAX_ROWS:
            raise HTTPException(413, f"At most {MAX_ROWS} rows in one request")
        try:
            return detector.update(rows)
        except ValueError as error:
            raise HTTPException(422, str(error))

    def demo_path(name: str) -> Path:
        return demo_dir / f"{name}.npz"

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models": len(models), "sessions": len(sessions)}

    @app.get("/api/models")
    def list_models():
        """The trained models, their settings, and if demo packets are available."""
        return [
            {
                "name": name,
                "title": TITLES.get(name, name),
                "n_features": model.n_features,
                "k_components": int(model.components.shape[0]),
                "settings": model.settings(),
                "trained_on": model.trained_on,
                "has_demo": demo_path(name).exists(),
            }
            for name, model in models.items()
        ]

    @app.post("/api/predict")
    def predict(request: PredictRequest):
        """Score a sequence of packets. The alert rule starts with no history.

        Use a session (``/api/sessions``) to keep the history between requests.
        """
        detector = get_detector(get_model(request.model), request.alert_rule, request.target_fpr)
        out = run(detector, request.rows)
        return _plain({
            "model": request.model, "alert_rule": request.alert_rule, "target_fpr": request.target_fpr,
            "threshold": detector.threshold, "scores": out["scores"], "values": out["values"],
            "alerts": out["alerts"], "alert_packets": detector.alert_packets, "alert_events": detector.alert_events,
        })

    @app.post("/api/sessions", status_code=201)
    def create_session(request: SessionRequest):
        """Start a detector that keeps its history, for a stream of packets."""
        detector = get_detector(get_model(request.model), request.alert_rule, request.target_fpr)
        while len(sessions) >= MAX_SESSIONS:
            sessions.popitem(last=False)
        session_id = uuid.uuid4().hex
        sessions[session_id] = detector
        return {"session_id": session_id, "model": request.model, "alert_rule": request.alert_rule,
                "target_fpr": request.target_fpr, "threshold": detector.threshold}

    @app.post("/api/sessions/{session_id}/packets")
    def send_packets(session_id: str, request: PacketsRequest):
        """Send the next packets of the stream. Returns the score and the alert of each packet."""
        if session_id not in sessions:
            raise HTTPException(404, "Unknown session")
        detector = sessions[session_id]
        out = run(detector, request.rows)
        return _plain({
            "scores": out["scores"], "values": out["values"], "alerts": out["alerts"],
            "packets_seen": detector.packets_seen, "alert_packets": detector.alert_packets,
            "alert_events": detector.alert_events,
        })

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        if sessions.pop(session_id, None) is None:
            raise HTTPException(404, "Unknown session")

    @app.get("/api/demo/{name}")
    def demo(name: str, alert_rule: str = "mean of 100", target_fpr: float = 1e-4):
        """Replay the recorded packets near the attack start through the detector."""
        model = get_model(name)
        if not demo_path(name).exists():
            raise HTTPException(404, f"No demo packets for '{name}'")
        data = np.load(demo_path(name))
        detector = get_detector(model, alert_rule, target_fpr)
        out = detector.update(data["features"])
        labels, start = data["labels"], int(data["attack_start"])
        summary = onset_evaluation(out["values"], labels, start, detector.threshold, detector.k, detector.n)
        return _plain({
            "model": name, "title": TITLES.get(name, name), "alert_rule": alert_rule, "target_fpr": target_fpr,
            "threshold": detector.threshold, "attack_start": start, "first_idx": int(data["first_idx"]),
            "values": np.round(out["values"], 6), "alerts": out["alerts"].astype(int), "labels": labels.astype(int),
            "summary": summary,
        })

    @app.get("/api/results")
    def results(alert_rule: str = "mean of 100", target_fpr: float = 1e-4):
        """Results of the full recordings (from the Kaggle run) for one setting."""
        if not results_csv.exists():
            raise HTTPException(404, "No results file")
        with open(results_csv, newline="") as fh:
            rows = [{k: _number(v) for k, v in row.items()} for row in csv.DictReader(fh)]
        rows = [r for r in rows if r["alert_rule"] == alert_rule and np.isclose(r["target_fpr"], target_fpr, rtol=1e-9)]
        for r in rows:
            r["title"] = TITLES.get(r["attack"], r["attack"])
        return rows

    return app


app = create_app()
