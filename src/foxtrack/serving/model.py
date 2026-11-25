"""The trained PCA detector and an online alert rule, in NumPy.

``PcaModel`` gives the same score as the Spark pipeline of the training run.
``OnlineDetector`` applies an alert rule to a stream of packets. It keeps the
last scores, so the packets can come in groups of any size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foxtrack.metrics import alerts_k_of_n, moving_mean


class PcaModel:
    def __init__(self, name: str, mean, std, components, thresholds: list[dict], trained_on: dict | None = None):
        self.name = name
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)  # (k, n_features)
        self.thresholds = thresholds
        self.trained_on = trained_on or {}
        self.source = {"type": "file"}   # where the service got this model
        if self.components.shape[1] != self.mean.size or self.std.size != self.mean.size:
            raise ValueError("The model arrays do not have the same number of features")

    @property
    def n_features(self) -> int:
        return self.mean.size

    @classmethod
    def from_dict(cls, d: dict) -> "PcaModel":
        return cls(d["name"], d["mean"], d["std"], d["components"], d["thresholds"], d.get("trained_on"))

    @classmethod
    def load(cls, path: str | Path) -> "PcaModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def score(self, features) -> np.ndarray:
        """Mean squared PCA reconstruction error of each row, in scaled units."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Each row must have {self.n_features} features")
        # The Spark scaler gives 0 for a feature that has no variance in the training data.
        z = np.divide(x - self.mean, self.std, out=np.zeros_like(x), where=self.std > 0)
        projected = z @ self.components.T
        error = (z * z).sum(axis=1) - (projected * projected).sum(axis=1)
        return np.maximum(error / self.n_features, 0.0)

    def rule(self, alert_rule: str, target_fpr: float) -> dict:
        for t in self.thresholds:
            if t["alert_rule"] == alert_rule and np.isclose(t["target_fpr"], target_fpr, rtol=1e-9):
                return t
        raise KeyError(f"No threshold for rule '{alert_rule}' at target rate {target_fpr}")

    def threshold(self, alert_rule: str, target_fpr: float) -> float:
        return self.rule(alert_rule, target_fpr)["threshold"]

    def settings(self) -> list[dict]:
        return [{"alert_rule": t["alert_rule"], "target_fpr": t["target_fpr"]} for t in self.thresholds]


class OnlineDetector:
    """One alert rule on a stream of packets.

    ``update`` returns, for each new packet: the score, the value that the
    rule compares with the threshold, and the alert. The result is the same
    as one call with all packets, for any group size.
    """

    def __init__(self, model: PcaModel, alert_rule: str = "mean of 100", target_fpr: float = 1e-4):
        self.model = model
        self.setting = model.rule(alert_rule, target_fpr)
        self.threshold = self.setting["threshold"]
        self.window = self.setting["smooth_window"]
        self.k, self.n = self.setting["k"], self.setting["n"]
        self._history = np.empty(0)   # the last scores that the rule still needs
        self.packets_seen = 0
        self.alert_packets = 0
        self.alert_events = 0
        self._last_alert_at = None    # position of the last packet with an alert

    def update(self, features, event_gap: int = 100) -> dict:
        scores = self.model.score(features)
        keep = max(self.window, self.n) - 1
        joined = np.concatenate((self._history, scores))
        if self.window > 1:
            values = moving_mean(joined, self.window)
            alerts = values > self.threshold
        else:
            values = joined
            alerts = alerts_k_of_n(joined, self.threshold, self.k, self.n)
        values, alerts = values[self._history.size:], alerts[self._history.size:]
        self._history = joined[-keep:] if keep else np.empty(0)

        for position in self.packets_seen + np.flatnonzero(alerts):
            if self._last_alert_at is None or position - self._last_alert_at > event_gap:
                self.alert_events += 1
            self._last_alert_at = int(position)
        self.packets_seen += scores.size
        self.alert_packets += int(alerts.sum())
        return {"scores": scores, "values": values, "alerts": alerts}
