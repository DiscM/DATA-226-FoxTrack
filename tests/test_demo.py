"""The demo builder must give the same packets as the Spark export of notebook 02."""

import json

import numpy as np
import pandas as pd

from foxtrack.demo import build_all, build_demo_window

N_ROWS, ONSET, N_FEATURES = 3000, 2000, 5


def _write_recording(folder, with_index_and_header):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    data = rng.random((N_ROWS, N_FEATURES)).round(5)
    data[:, 0] = np.arange(N_ROWS)  # the row position, to check the window
    labels = (np.arange(N_ROWS) >= ONSET).astype(int)
    pd.DataFrame(data).to_csv(folder / "X_dataset.csv", index=with_index_and_header, header=with_index_and_header)
    pd.DataFrame({"x": labels}).to_csv(folder / "X_labels.csv", index=True)
    return data, labels


def test_window_position_and_content(tmp_path):
    data, labels = _write_recording(tmp_path / "OS Scan", with_index_and_header=False)
    w = build_demo_window(tmp_path / "OS Scan" / "X_dataset.csv", tmp_path / "OS Scan" / "X_labels.csv",
                          first_attack_idx=ONSET, calib_end=1700, before=500, after=400)
    # The window cannot start before the calibration end.
    assert int(w["first_idx"]) == 1700 and int(w["attack_start"]) == 300
    assert w["features"].shape == (700, N_FEATURES) and w["features"].dtype == np.float32
    assert w["features"][:, 0].tolist() == list(range(1700, 2400))
    assert np.allclose(w["features"], data[1700:2400], atol=1e-6)
    assert w["labels"].tolist() == labels[1700:2400].tolist()


def test_file_with_index_column_and_header(tmp_path):
    data, _ = _write_recording(tmp_path / "Mirai Botnet", with_index_and_header=True)
    w = build_demo_window(tmp_path / "Mirai Botnet" / "X_dataset.csv", tmp_path / "Mirai Botnet" / "X_labels.csv",
                          first_attack_idx=ONSET, calib_end=0, before=100, after=100)
    assert w["features"].shape == (200, N_FEATURES)
    assert w["features"][:, 0].tolist() == list(range(1900, 2100))


def test_build_all_reports_missing_recordings(tmp_path):
    _write_recording(tmp_path / "raw" / "OS Scan", with_index_and_header=False)
    models = tmp_path / "models"; models.mkdir()
    for name in ("os_scan", "mirai_botnet"):
        (models / f"{name}.json").write_text(json.dumps({"trained_on": {"first_attack_idx": ONSET, "calib_end": 1700}}))
    result = build_all(tmp_path / "raw", models, tmp_path / "demo")
    assert result == {"built": ["os_scan"], "missing": ["mirai_botnet"]}
    saved = np.load(tmp_path / "demo" / "os_scan.npz")
    assert int(saved["attack_start"]) == 300 and saved["features"].shape[0] == N_ROWS - 1700
