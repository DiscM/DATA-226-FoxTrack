"""Build the demo packets for the dashboard from the Kitsune CSV files.

The demo packets are a part of the Kitsune dataset, so they are not in this
repository. This module reads them from dataset files that you download from
Kaggle. It needs pandas and NumPy only, not Spark.

The result is equal to the `demo/` output of notebook 02.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from foxtrack import etl

BEFORE, AFTER = 4000, 6000


def build_demo_window(features_path, labels_path, first_attack_idx: int, calib_end: int,
                      before: int = BEFORE, after: int = AFTER) -> dict:
    """Read the packets near the attack start of one recording."""
    features_path, labels_path = Path(features_path), Path(labels_path)
    info = etl.sniff_features(features_path)
    start = max(first_attack_idx - before, calib_end)
    stop = first_attack_idx + after
    rows = pd.read_csv(
        features_path, header=None, skiprows=start + (1 if info["has_header"] else 0), nrows=stop - start,
    )
    if info["has_index"]:
        rows = rows.iloc[:, 1:]
    labels = etl.load_labels(labels_path)[start:stop]
    if len(rows) != len(labels):
        raise ValueError(f"{features_path.name}: {len(rows)} feature rows but {len(labels)} labels in the window")
    return {
        "features": rows.to_numpy(dtype=np.float32),
        "labels": labels.astype(np.int8),
        "first_idx": np.int64(start),
        "attack_start": np.int64(first_attack_idx - start),
    }


def build_all(kitsune_dir, models_dir, out_dir) -> dict:
    """Build the demo file for each recording that is in ``kitsune_dir``.

    Returns {"built": [...], "missing": [...]} with the names of the recordings.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captures = {c.name: c for c in etl.discover(kitsune_dir)}
    built, missing = [], []
    for model_path in sorted(Path(models_dir).glob("*.json")):
        name = model_path.stem
        if name not in captures:
            missing.append(name)
            continue
        trained = json.loads(model_path.read_text())["trained_on"]
        window = build_demo_window(captures[name].features_path, captures[name].labels_path,
                                   trained["first_attack_idx"], trained["calib_end"])
        np.savez_compressed(out_dir / f"{name}.npz", **window)
        built.append(name)
    return {"built": built, "missing": missing}
