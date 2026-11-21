"""Baseline anomaly detector: PCA reconstruction error, trained on benign traffic.

Spark MLlib fits a scaler and a PCA on benign packets. The score of a packet
is the part of it that the PCA cannot rebuild. Traffic that does not look like
the benign training traffic gets a high score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foxtrack.metrics import onset_evaluation
from foxtrack.split import OnsetSplit


def feature_columns(df) -> list[str]:
    return sorted((c for c in df.columns if c[0] == "f" and c[1:].isdigit()), key=lambda c: int(c[1:]))


def fit_pca_detector(train_df, k: int = 20):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler

    cols = feature_columns(train_df)
    return Pipeline(stages=[
        VectorAssembler(inputCols=cols, outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="scaled", withMean=True, withStd=True),
        PCA(inputCol="scaled", outputCol="pca", k=k),
    ]).fit(train_df)


def score(model, df):
    """Add ``score``: mean squared reconstruction error in scaled units.

    The PCA axes are orthonormal, so the error is |x|^2 - |projection|^2.
    This runs in the JVM, with no Python code for each row.
    """
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    def sum_sq(col):
        return F.aggregate(vector_to_array(col), F.lit(0.0), lambda acc, v: acc + v * v)

    n_features = len(feature_columns(df))
    out = model.transform(df)
    err = (sum_sq(F.col("scaled")) - sum_sq(F.col("pca"))) / n_features
    return out.select("idx", "label", F.greatest(err, F.lit(0.0)).alias("score"))


def run_capture(
    capture_df,
    split: OnsetSplit,
    k_components: int = 20,
    target_fprs=(1e-2, 1e-3, 1e-4),
    alert_rules=((1, 1), (3, 5), (10, 20)),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on benign packets, set thresholds, and measure the detection delay.

    Returns (results, test_scores). ``results`` has one row for each
    combination of target false positive rate and alert rule.
    """
    from pyspark.sql import functions as F

    idx = F.col("idx")
    model = fit_pca_detector(capture_df.where(idx < split.train_end), k_components)

    calib = score(model, capture_df.where((idx >= split.train_end) & (idx < split.calib_end)))
    calib_scores = calib.select("score").toPandas()["score"].to_numpy()

    test = score(model, capture_df.where(idx >= split.calib_end)).toPandas().sort_values("idx")
    scores = test["score"].to_numpy()
    labels = test["label"].to_numpy()

    rows = []
    for fpr in target_fprs:
        threshold = float(np.quantile(calib_scores, 1 - fpr))
        for k, n in alert_rules:
            result = onset_evaluation(scores, labels, split.onset_in_test, threshold, k, n)
            rows.append({"attack": split.attack, "target_fpr": fpr, **result})
    return pd.DataFrame(rows), test
