"""Baseline detector test on synthetic traffic with a known attack start."""

import numpy as np
import pandas as pd
import pytest

from foxtrack import anomaly, etl
from foxtrack.metrics import onset_evaluation
from foxtrack.split import onset_split

N_ROWS, ONSET, N_FEATURES = 6000, 4000, 12


@pytest.fixture(scope="module")
def spark():
    session = etl.build_spark("foxtrack-test", driver_memory="1g")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def capture(spark):
    rng = np.random.default_rng(0)
    # Benign traffic lies near a 3-dimension plane. The attack leaves the plane.
    latent = rng.normal(size=(N_ROWS, 3))
    data = latent @ rng.normal(size=(3, N_FEATURES)) + 0.05 * rng.normal(size=(N_ROWS, N_FEATURES))
    data[ONSET:] += rng.normal(scale=3.0, size=(N_ROWS - ONSET, N_FEATURES))
    df = pd.DataFrame(data.astype("float32"), columns=[f"f{i}" for i in range(N_FEATURES)])
    df.insert(0, "label", (np.arange(N_ROWS) >= ONSET).astype("int8"))
    df.insert(0, "idx", np.arange(N_ROWS, dtype="int64"))
    return spark.createDataFrame(df)


def test_onset_split_ranges():
    split = onset_split({"attack": "x", "first_attack_idx": 1000, "rows": 1500})
    assert (split.train_end, split.calib_end, split.onset_in_test) == (700, 850, 150)
    with pytest.raises(ValueError):
        onset_split({"attack": "x", "first_attack_idx": 0, "rows": 10})


def test_onset_evaluation():
    scores = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1], dtype=float)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    out = onset_evaluation(scores, labels, onset=5, threshold=0.5)
    assert out["delay_packets"] == 2
    assert out["false_alarms_before_attack"] == 1
    assert out["packet_recall"] == pytest.approx(0.6)
    missed = onset_evaluation(np.zeros(10), labels, onset=5, threshold=0.5)
    assert missed["detected"] is False and missed["delay_packets"] is None


def test_feature_columns_sequence(capture):
    assert anomaly.feature_columns(capture) == [f"f{i}" for i in range(N_FEATURES)]


def test_score_agrees_with_numpy(capture):
    model = anomaly.fit_pca_detector(capture.where("idx < 2800"), k=3)
    got = anomaly.score(model, capture).toPandas().sort_values("idx")["score"].to_numpy()

    data = capture.toPandas().sort_values("idx")[[f"f{i}" for i in range(N_FEATURES)]].to_numpy(dtype="float64")
    train = data[:2800]
    z = (data - train.mean(0)) / train.std(0, ddof=1)
    _, _, vt = np.linalg.svd((train - train.mean(0)) / train.std(0, ddof=1), full_matrices=False)
    expected = ((z - z @ vt[:3].T @ vt[:3]) ** 2).sum(1) / N_FEATURES
    assert np.allclose(got, expected, rtol=1e-3, atol=1e-6)


def test_run_capture_detects_attack_start(capture):
    split = onset_split({"attack": "synthetic", "first_attack_idx": ONSET, "rows": N_ROWS})
    results, test = anomaly.run_capture(capture, split, k_components=3, target_fprs=(1e-2,), alert_rules=((1, 1), (3, 5)))
    assert len(test) == N_ROWS - split.calib_end
    assert results["detected"].all()
    assert (results["delay_packets"] <= 5).all()
    assert (results["false_alarms_per_100k"] < 5000).all()
