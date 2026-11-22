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

import pandas as pd

from foxtrack.anomaly import feature_columns
from foxtrack.metrics import evaluate_grid, ranking_metrics
from foxtrack.split import OnsetSplit


def build_training_sample(
    spark, parquet_root, summary: list[dict], n_per_class: int = 50_000, seed: int = 7,
    benign_period: str = "all", min_benign: int = 1000,
):
    """Random sample of each capture: at most ``n_per_class`` benign and malicious packets.

    A sample is necessary. Gradient-boosted trees on all 21 million packets
    are too slow for one machine, and the classes are far from balanced.

    benign_period="all"     benign packets from the full capture.
    benign_period="attack"  benign packets only from the attack period. Benign
        and malicious packets then come from the same time in the capture, so
        the model cannot use the time as a sign of an attack. A capture with
        fewer than ``min_benign`` benign packets in the attack period is not used.
    """
    from pyspark.sql import functions as F

    if benign_period not in ("all", "attack"):
        raise ValueError("benign_period must be 'all' or 'attack'")
    parts = []
    for row in summary:
        df = spark.read.parquet(f"{parquet_root}/features/attack={row['attack']}")
        benign = df.where(F.col("label") == 0)
        n_benign = row["rows"] - row["malicious"]
        if benign_period == "attack":
            benign = benign.where(F.col("idx") >= row["first_attack_idx"])
            n_benign = benign.count()
            if n_benign < min_benign:
                continue
        for part, count in ((benign, n_benign), (df.where(F.col("label") == 1), row["malicious"])):
            if count == 0:
                continue
            if count > n_per_class:
                # A small margin, then a limit, gives almost exactly n_per_class rows.
                part = part.sample(fraction=min(1.0, 1.1 * n_per_class / count), seed=seed).limit(n_per_class)
            parts.append(part.withColumn("attack", F.lit(row["attack"])))
    if not parts:
        raise ValueError("No capture has sufficient benign packets for the training sample")
    out = parts[0]
    for part in parts[1:]:
        out = out.unionByName(part)
    return out


def build_clock_sample(spark, parquet_root, summary: list[dict], n_per_class: int = 50_000, seed: int = 7, late_frac: float = 0.15):
    """Benign packets before the attack, with a label for their time in the capture.

    label = 1 for the last ``late_frac`` of the benign period, 0 for the
    packets before that. No attack packet is in this sample. A model that can
    predict this label has learned a clock, not an attack.
    """
    from pyspark.sql import functions as F

    parts = []
    for row in summary:
        onset = row["first_attack_idx"]
        cut = int(onset * (1 - late_frac))
        df = spark.read.parquet(f"{parquet_root}/features/attack={row['attack']}").where(F.col("idx") < onset)
        for label, part, count in ((0, df.where(F.col("idx") < cut), cut), (1, df.where(F.col("idx") >= cut), onset - cut)):
            if count > n_per_class:
                part = part.sample(fraction=min(1.0, 1.1 * n_per_class / count), seed=seed).limit(n_per_class)
            parts.append(part.withColumn("label", F.lit(label).cast("byte")).withColumn("attack", F.lit(row["attack"])))
    out = parts[0]
    for part in parts[1:]:
        out = out.unionByName(part)
    return out


def assemble(df, features: list[str] | None = None):
    """Put the feature columns (all, or only ``features``) into one vector column."""
    from pyspark.ml.feature import VectorAssembler

    keep = [c for c in ("attack", "idx", "label") if c in df.columns]
    cols = features or feature_columns(df)
    return VectorAssembler(inputCols=cols, outputCol="features").transform(df).select(*keep, "features")


def feature_importance(model, features: list[str]) -> pd.Series:
    """Importance of each feature in a tree model, largest first. The sum is 1."""
    values = model.featureImportances.toArray()
    return pd.Series(values, index=features, name="importance").sort_values(ascending=False)


def top_features(importance: pd.Series, share: float = 0.8, max_features: int = 40) -> list[str]:
    """The smallest set of features that has ``share`` of the total importance."""
    cumulative = importance.cumsum()
    n = int((cumulative < share).sum()) + 1
    return importance.index[: min(n, max_features)].tolist()


def clock_fold_auc(clock_assembled, attack: str, **params) -> float:
    """Train the clock model on all captures but one. ROC-AUC on that capture."""
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from sklearn.metrics import roc_auc_score

    model = make_estimator("gbt", **params).fit(clock_assembled.where(F.col("attack") != attack))
    held = model.transform(clock_assembled.where(F.col("attack") == attack))
    out = held.select("label", vector_to_array("probability")[1].alias("score")).toPandas()
    return float(roc_auc_score(out["label"], out["score"]))


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
    features: list[str] | None = None,
    smooth_windows=(1, 100),
    **params,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all captures but ``split.attack``, then test on that capture.

    ``train_assembled`` is the output of ``assemble`` on the training sample
    of ALL captures. This function removes the test capture from it.
    ``features`` must be the same list that ``assemble`` got for the sample.
    """
    from pyspark.sql import functions as F

    train = train_assembled.where(F.col("attack") != split.attack)
    model = make_estimator(model_name, **params).fit(train)

    idx = F.col("idx")
    held = assemble(capture_df.where(idx >= split.train_end), features)
    scored = score(model, held).toPandas().sort_values("idx")
    is_calib = scored["idx"].to_numpy() < split.calib_end
    test = scored[~is_calib]
    scores = test["score"].to_numpy()
    labels = test["label"].to_numpy()

    ranking = ranking_metrics(scores, labels)
    grid = evaluate_grid(
        scored["score"].to_numpy()[is_calib], scores, labels,
        split.onset_in_test, target_fprs, alert_rules, smooth_windows,
    )
    rows = [{"attack": split.attack, "model": model_name, **row, **ranking} for row in grid]
    return pd.DataFrame(rows), test
