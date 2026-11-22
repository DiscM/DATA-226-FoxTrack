"""Scoring code for the service. It needs NumPy only, not Spark."""

from foxtrack.serving.model import OnlineDetector, PcaModel

__all__ = ["OnlineDetector", "PcaModel"]
