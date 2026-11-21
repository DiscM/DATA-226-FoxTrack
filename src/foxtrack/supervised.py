"""Supervised models with a leave-one-capture-out test.

Each capture has one attack start, so a supervised model cannot train and
test on the same capture and still measure the delay at that start. Here the
model trains on labelled packets from all captures but one, and the test is
the capture that it did not see. The test thus shows the detection of an
attack type that is new to the model.

The alert threshold comes from benign packets of the test capture, the same
as for the baseline. The results are thus comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foxtrack.anomaly import feature_columns
from foxtrack.metrics import onset_evaluation, ranking_metrics
from foxtrack.split import OnsetSplit


def build_training_sample(spark, parquet_root, summary: list[dict], n_per_class: int = 50_000, seed: int = 7):
    """Random sample of each capture: at most ``n_per_class`` benign and malicious packets.

    A sample is necessary. Gradient-boosted trees on all 21 million packets
    are too slow for one machine, and the classes are far from balanced.
    """
    from pyspark.sql import functions as F

    parts = []
    for row in summary:
        df = spark.read.parquet(f"{parquet_root}/features/attack={row['attack']}")
        counts = {1: row["malicious"], 0: row["rows"] - row["malicious"]}
        for label, count in counts.items():
            if count == 0:
                continue
            part = df.where(F.col("label") == label)
            if count > n_per_class:
                # A small margin, then a limit, gives almost exactly n_per_class rows.
                part = part.sample(fraction=min(1.0, 1.1 * n_per_class / count), seed=seed).limit(n_per_class)
            parts.append(part.withColumn("attack", F.lit(row["attack"])))
    out = parts[0]
    for part in parts[1:]:
        out = out.unionByName(part)
    return out


def assemble(df):
    """Put the feature columns into one vector column, ``features``."""
    from pyspark.ml.feature import VectorAssembler

    keep = [c for c in ("attack", "idx", "label") if c in df.columns]
    return VectorAssembler(inputCols=feature_columns(df), outputCol="features").transform(df).select(*keep, "features")


def make_estimator(name: str, seed: int = 7, **params):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.feature import StandardScaler

    if name == "logreg":
        settings = {"maxIter": 50, "regParam": 0.01, **params}
        return Pipeline(stages=[
            StandardScaler(inputCol="features", outputCol="scaled", withMean=True, withStd=True),
            LogisticRegression(featuresCol="scaled", labelCol="label", **settings),
        ])
    if name == "gbt":
        settings = {"maxIter": 30, "maxDepth": 5, "stepSize": 0.1, "subsamplingRate": 0.8, "maxBins": 64, **params}
        return GBTClassifier(featuresCol="features", labelCol="label", seed=seed, **settings)
    raise ValueError(f"Unknown model: {name}")


def score(model, assembled_df):
    """Add ``score``: the probability that the packet is malicious."""
    from pyspark.ml.functions import vector_to_array

    out = model.transform(assembled_df)
    return out.select("idx", "label", vector_to_array("probability")[1].alias("score"))


def run_fold(
    train_assembled,
    capture_df,
    split: OnsetSplit,
    model_name: str,
    target_fprs=(1e-2, 1e-3, 1e-4),
    alert_rules=((1, 1), (3, 5), (10, 20)),
    **params,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all captures but ``split.attack``, then test on that capture.

    ``train_assembled`` is the output of ``assemble`` on the training sample
    of ALL captures. This function removes the test capture from it.
    """
    from pyspark.sql import functions as F

    train = train_assembled.where(F.col("attack") != split.attack)
    model = make_estimator(model_name, **params).fit(train)

    idx = F.col("idx")
    held = assemble(capture_df.where(idx >= split.train_end))
    scored = score(model, held).toPandas().sort_values("idx")
    is_calib = scored["idx"].to_numpy() < split.calib_end
    calib_scores = scored["score"].to_numpy()[is_calib]
    test = scored[~is_calib]
    scores = test["score"].to_numpy()
    labels = test["label"].to_numpy()

    ranking = ranking_metrics(scores, labels)
    rows = []
    for fpr in target_fprs:
        threshold = float(np.quantile(calib_scores, 1 - fpr))
        for k, n in alert_rules:
            result = onset_evaluation(scores, labels, split.onset_in_test, threshold, k, n)
            rows.append({"attack": split.attack, "model": model_name, "target_fpr": fpr, **result, **ranking})
    return pd.DataFrame(rows), test
