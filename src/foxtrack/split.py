"""Train / calibration / test ranges for each capture.

Each Kitsune capture starts with a long benign period, and the attack starts
one time. A random split or a plain time split thus cannot test the detection
delay with a supervised model: the attack start would be in the training data.

The onset split uses only benign packets for training:

    [0, train_end)          benign, to fit the model
    [train_end, calib_end)  benign, to set the alert threshold
    [calib_end, rows)       test: benign packets, then the attack start
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OnsetSplit:
    attack: str
    train_end: int
    calib_end: int
    onset: int
    rows: int

    @property
    def onset_in_test(self) -> int:
        """Position of the attack start in the test range."""
        return self.onset - self.calib_end


def load_summary(parquet_root: str | Path) -> list[dict]:
    return json.loads((Path(parquet_root) / "summary.json").read_text())


def onset_split(row: dict, train_frac: float = 0.70, calib_frac: float = 0.15) -> OnsetSplit:
    """Divide the benign period before the first attack packet."""
    if not 0 < train_frac < train_frac + calib_frac < 1:
        raise ValueError("train_frac and calib_frac must leave benign packets for the test")
    onset = row["first_attack_idx"]
    if not onset:
        raise ValueError(f"{row['attack']}: no benign period before the attack")
    train_end = int(onset * train_frac)
    calib_end = int(onset * (train_frac + calib_frac))
    return OnsetSplit(row["attack"], train_end, calib_end, onset, row["rows"])
