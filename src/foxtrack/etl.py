"""Convert the Kitsune CSV files to Parquet with Spark.

Each capture has a feature file (one row for each packet, no label) and a
label file. The two files agree only by row position. This module gives each
feature row its position (``idx``) and joins the label on that position.

Output layout:
    <out>/features/attack=<name>/*.parquet   idx, label, f0 .. f114
    <out>/summary.json                       row counts and attack episodes
"""

from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from foxtrack.metrics import find_episodes

EXPECTED_FEATURES = 115

# Spark divides a large plain CSV into one part for each CPU core. Row position
# is reliable only if each file is one partition. Both settings are necessary:
# Spark uses min(maxPartitionBytes, max(openCostInBytes, total bytes / cores)).
ONE_PARTITION_BYTES = str(256 * 1024**3)


@dataclass
class Capture:
    name: str
    features_path: Path
    labels_path: Path


def build_spark(app_name: str = "foxtrack", driver_memory: str = "20g"):
    """Spark session for one machine (Kaggle Notebook or laptop)."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def discover(root: str | Path) -> list[Capture]:
    """Find feature/label file pairs below ``root``. One pair for each directory."""
    root = Path(root)
    captures = []
    for feat in sorted(root.rglob("*")):
        if not feat.is_file() or not re.search(r"dataset.*\.csv(\.gz)?$", feat.name, re.I):
            continue
        labels = [
            p for p in feat.parent.iterdir()
            if p.is_file() and re.search(r"label.*\.csv(\.gz)?$", p.name, re.I)
        ]
        if len(labels) != 1:
            raise FileNotFoundError(f"Expected one label file adjacent to {feat}, found {len(labels)}")
        captures.append(Capture(_slug(feat.parent.name), feat, labels[0]))
    if not captures:
        raise FileNotFoundError(f"No '*dataset*.csv' files below {root}")
    return captures


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def sniff_features(path: Path, sample_rows: int = 50) -> dict:
    """Read the first rows to find the header, column count, and an index column."""
    with _open_text(path) as fh:
        lines = [line.rstrip("\n") for _, line in zip(range(sample_rows + 1), fh)]
    first = lines[0].split(",")
    named = [x for x in first if x != ""]
    # A header is text, or column numbers such as 0,1,2,... (pandas default).
    numbered = (
        len(named) > 2
        and all(re.fullmatch(r"\d+", x) for x in named)
        and all(int(b) - int(a) == 1 for a, b in zip(named, named[1:]))
    )
    has_header = numbered or not all(_is_number(x) for x in named)
    data = [line.split(",") for line in (lines[1:] if has_header else lines) if line]
    n_cols = len(data[0])
    # A first column of 0,1,2,... or 1,2,3,... is a row index, not a feature.
    col0 = [row[0] for row in data]
    has_index = (
        len(col0) >= 3
        and all(re.fullmatch(r"\d+", x) for x in col0)
        and all(int(b) - int(a) == 1 for a, b in zip(col0, col0[1:]))
        and int(col0[0]) in (0, 1)
    )
    return {"has_header": has_header, "n_cols": n_cols, "has_index": has_index}


def load_labels(path: Path) -> np.ndarray:
    """Read a label file to a 0/1 int8 array.

    The files are not uniform: some have a header, and some have an index
    column. The label is always the last column.
    """
    df = pd.read_csv(path, header=None, dtype=str)
    if not _is_number(df.iloc[0, -1]):
        df = df.iloc[1:]
    labels = pd.to_numeric(df.iloc[:, -1]).to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{path}: labels are not all 0 or 1")
    return labels.astype(np.int8)


def convert_capture(spark, cap: Capture, out_dir: str | Path, merge_gap: int = 1000) -> dict:
    """Convert one capture to Parquet. Returns a summary dict."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import FloatType, StructField, StructType

    out_dir = Path(out_dir)
    info = sniff_features(cap.features_path)
    n_feat = info["n_cols"] - (1 if info["has_index"] else 0)

    names = (["_row"] if info["has_index"] else []) + [f"f{i}" for i in range(n_feat)]
    schema = StructType([StructField(n, FloatType(), True) for n in names])
    feats = spark.read.csv(str(cap.features_path), schema=schema, header=info["has_header"])
    if info["has_index"]:
        feats = feats.drop("_row")
    if feats.rdd.getNumPartitions() != 1:
        raise RuntimeError(f"{cap.name}: more than one partition, row position is not reliable")
    feats = feats.withColumn("idx", F.monotonically_increasing_id())

    labels = load_labels(cap.labels_path)
    labels_file = out_dir / "_labels" / f"{cap.name}.parquet"
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"idx": np.arange(labels.size, dtype=np.int64), "label": labels}).to_parquet(labels_file)
    labels_df = spark.read.parquet(str(labels_file))

    target = out_dir / "features" / f"attack={cap.name}"
    joined = feats.join(F.broadcast(labels_df), "idx", "left")
    joined.select("idx", "label", *[f"f{i}" for i in range(n_feat)]).write.mode("overwrite").parquet(str(target))

    # Check the result from Parquet. This is much faster than a second CSV read.
    check = spark.read.parquet(str(target)).agg(
        F.count("*").alias("rows"),
        F.sum(F.col("label").isNull().cast("int")).alias("no_label"),
        F.sum(F.col("f0").isNull().cast("int")).alias("null_f0"),
    ).first()
    if check["rows"] != labels.size or check["no_label"]:
        raise RuntimeError(
            f"{cap.name}: {check['rows']} feature rows but {labels.size} labels "
            f"({check['no_label']} rows without a label)"
        )

    episodes = find_episodes(labels, merge_gap)
    n_mal = int(labels.sum())
    return {
        "attack": cap.name,
        "rows": int(labels.size),
        "features": n_feat,
        "features_as_expected": n_feat == EXPECTED_FEATURES,
        "dropped_index_column": info["has_index"],
        "null_f0_rows": int(check["null_f0"] or 0),
        "malicious": n_mal,
        "malicious_fraction": n_mal / labels.size,
        "first_attack_idx": int(episodes[0, 0]) if len(episodes) else None,
        "raw_runs": int(len(find_episodes(labels))),
        "episodes": int(len(episodes)),
        "episode_bounds": episodes[:20].tolist(),
    }


def convert_all(spark, root: str | Path, out_dir: str | Path, workers: int = 4, merge_gap: int = 1000) -> pd.DataFrame:
    """Convert all captures. Each capture is one Spark task, so run them in parallel."""
    captures = discover(root)
    keys = ("spark.sql.files.maxPartitionBytes", "spark.sql.files.openCostInBytes")
    before = {k: spark.conf.get(k) for k in keys}
    for k in keys:
        spark.conf.set(k, ONE_PARTITION_BYTES)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda c: convert_capture(spark, c, out_dir, merge_gap), captures))
    finally:
        # Other work (EDA, training) must read files in parallel again.
        for k in keys:
            spark.conf.set(k, before[k])
    out_dir = Path(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    return pd.DataFrame(rows).drop(columns=["episode_bounds"])
