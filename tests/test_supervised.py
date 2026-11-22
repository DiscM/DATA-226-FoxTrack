"""Leave-one-capture-out test on synthetic captures that share one attack signal."""

import json

import numpy as np
import pandas as pd
import pytest

from foxtrack import etl, supervised
from foxtrack.split import load_summary, onset_split

N_ROWS, ONSET, N_FEATURES = 3000, 2000, 8
NAMES = ["alpha", "beta", "gamma"]


@pytest.fixture(scope="module")
def spark():
    session = etl.build_spark("foxtrack-test", driver_memory="1g")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def parquet_root(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("parquet")
    summary = []
    for i, name in enumerate(NAMES):
        rng = np.random.default_rng(i)
        data = rng.normal(size=(N_ROWS, N_FEATURES))
        data[ONSET:, 0] += 6.0  # the attack signal, the same in all captures
        df = pd.DataFrame(data.astype("float32"), columns=[f"f{j}" for j in range(N_FEATURES)])
        df.insert(0, "label", (np.arange(N_ROWS) >= ONSET).astype("int8"))
        df.insert(0, "idx", np.arange(N_ROWS, dtype="int64"))
        spark.createDataFrame(df).write.parquet(str(root / "features" / f"attack={name}"))
        summary.append({"attack": name, "rows": N_ROWS, "malicious": N_ROWS - ONSET, "first_attack_idx": ONSET})
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def test_training_sample_is_limited_for_each_class(spark, parquet_root):
    sample = supervised.build_training_sample(spark, parquet_root, load_summary(parquet_root), n_per_class=300)
    counts = sample.groupBy("attack", "label").count().toPandas()
    assert len(counts) == 6
    assert counts["count"].between(250, 300).all()


@pytest.mark.parametrize("model_name", ["logreg", "gbt"])
def test_run_fold_detects_unseen_capture(spark, parquet_root, model_name):
    summary = load_summary(parquet_root)
    sample = supervised.build_training_sample(spark, parquet_root, summary, n_per_class=500)
    train = supervised.assemble(sample).cache()
    split = onset_split(summary[0])
    capture = spark.read.parquet(str(parquet_root / "features" / f"attack={split.attack}"))

    params = {"maxIter": 5} if model_name == "gbt" else {}
    results, test = supervised.run_fold(
        train, capture, split, model_name, target_fprs=(1e-2,), alert_rules=((1, 1), (3, 5)), **params
    )
    assert len(test) == N_ROWS - split.calib_end
    assert test["idx"].min() == split.calib_end
    assert results["detected"].all()
    assert (results["delay_packets"] <= 5).all()
    assert (results["roc_auc"] > 0.95).all()
    assert set(results["model"]) == {model_name}


def test_unknown_model():
    with pytest.raises(ValueError):
        supervised.make_estimator("svm")


# ---- Time confound test: captures with a clock feature ----

@pytest.fixture(scope="module")
def clock_root(spark, tmp_path_factory):
    """f0 = weak attack signal. f1 = a clock: it changes a short time before the attack and stays changed."""
    root = tmp_path_factory.mktemp("clock")
    summary = []
    for i, name in enumerate(NAMES):
        rng = np.random.default_rng(10 + i)
        data = rng.normal(size=(N_ROWS, N_FEATURES))
        malicious = (np.arange(N_ROWS) >= ONSET) & (rng.random(N_ROWS) < 0.5)
        data[malicious, 0] += 2.0
        data[int(ONSET * 0.9):, 1] += 6.0
        df = pd.DataFrame(data.astype("float32"), columns=[f"f{j}" for j in range(N_FEATURES)])
        df.insert(0, "label", malicious.astype("int8"))
        df.insert(0, "idx", np.arange(N_ROWS, dtype="int64"))
        spark.createDataFrame(df).write.parquet(str(root / "features" / f"attack={name}"))
        summary.append({"attack": name, "rows": N_ROWS, "malicious": int(malicious.sum()), "first_attack_idx": ONSET})
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def test_clock_model_finds_the_clock_feature(spark, clock_root):
    summary = load_summary(clock_root)
    sample = supervised.build_clock_sample(spark, clock_root, summary, n_per_class=400)
    assert sample.where("idx >= %d" % ONSET).count() == 0  # no packet from the attack period
    clock = supervised.assemble(sample).cache()
    assert supervised.clock_fold_auc(clock, "alpha", maxIter=5) > 0.75

    features = [f"f{j}" for j in range(N_FEATURES)]
    model = supervised.make_estimator("gbt", maxIter=5).fit(clock)
    importance = supervised.feature_importance(model, features)
    assert importance.index[0] == "f1"
    assert importance.sum() == pytest.approx(1.0)
    assert supervised.top_features(importance, share=0.5) == ["f1"]


def test_time_matched_sample_has_benign_only_from_attack_period(spark, clock_root):
    summary = load_summary(clock_root)
    sample = supervised.build_training_sample(spark, clock_root, summary, n_per_class=300, benign_period="attack", min_benign=100)
    assert sample.where("label = 0 and idx < %d" % ONSET).count() == 0
    assert sample.where("label = 0").count() > 600
    with pytest.raises(ValueError):
        supervised.build_training_sample(spark, clock_root, summary, benign_period="late")


def _alert_rate_before_attack(spark, root, variant):
    """Alert rate on benign packets before the attack, with a threshold from early benign packets."""
    summary = load_summary(root)
    features = [f"f{j}" for j in range(N_FEATURES)]
    period = "attack" if variant == "time_matched" else "all"
    if variant == "drop_clock":
        features = [f for f in features if f != "f1"]
    sample = supervised.build_training_sample(spark, root, summary, n_per_class=500, benign_period=period, min_benign=100)
    train = supervised.assemble(sample, features)
    split = onset_split(summary[0])
    capture = spark.read.parquet(str(root / "features" / f"attack={split.attack}"))
    results, _ = supervised.run_fold(train, capture, split, "gbt", target_fprs=(1e-2,), alert_rules=((1, 1),),
                                     features=features, maxIter=10)
    return results["false_alarms_per_100k"].iloc[0] / 1000  # percent


def test_clock_confound_and_its_two_corrections(spark, clock_root):
    confounded = _alert_rate_before_attack(spark, clock_root, "standard")
    assert confounded > 15  # the model gives alerts on late benign packets
    assert _alert_rate_before_attack(spark, clock_root, "drop_clock") < confounded / 3
    assert _alert_rate_before_attack(spark, clock_root, "time_matched") < confounded / 3
