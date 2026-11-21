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
