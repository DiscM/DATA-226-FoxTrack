"""ETL test on small synthetic files that copy the known Kitsune file variations."""

import gzip

import numpy as np
import pandas as pd
import pytest

from foxtrack import etl

N_FEATURES = 115


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)


def _features(n_rows, seed):
    # Column 0 holds the row position, so the test can check the row sequence.
    data = np.random.default_rng(seed).random((n_rows, N_FEATURES)).round(4)
    data[:, 0] = np.arange(n_rows)
    return data


def _csv(data, index=False, header=False):
    return pd.DataFrame(data).to_csv(index=index, header=header)


@pytest.fixture(scope="module")
def spark():
    session = etl.build_spark("foxtrack-test", driver_memory="1g")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("kitsune")
    labels_a = np.array([0] * 300 + [1] * 200)
    labels_b = np.array([0] * 100 + [1] * 50 + [0] * 100 + [1] * 50)

    # Variation 1: gzip, no header on features, labels with index column and header.
    _write(root / "ARP MitM" / "ARP_MitM_dataset.csv.gz", _csv(_features(500, 1)))
    _write(
        root / "ARP MitM" / "ARP_MitM_labels.csv.gz",
        pd.DataFrame({"x": labels_a}).to_csv(index=True),
    )
    # Variation 2: plain CSV, features with index column and header, bare labels.
    _write(root / "Mirai Botnet" / "Mirai_dataset.csv", _csv(_features(300, 2), index=True, header=True))
    _write(root / "Mirai Botnet" / "Mirai_labels.csv", "\n".join(map(str, labels_b)) + "\n")
    return root, {"arp_mitm": labels_a, "mirai_botnet": labels_b}


def test_discover(raw):
    root, _ = raw
    assert [c.name for c in etl.discover(root)] == ["arp_mitm", "mirai_botnet"]


def test_sniff_detects_index_and_header(raw):
    root, _ = raw
    plain = etl.sniff_features(root / "ARP MitM" / "ARP_MitM_dataset.csv.gz")
    assert plain == {"has_header": False, "n_cols": N_FEATURES, "has_index": False}
    indexed = etl.sniff_features(root / "Mirai Botnet" / "Mirai_dataset.csv")
    assert indexed == {"has_header": True, "n_cols": N_FEATURES + 1, "has_index": True}


def test_convert_all(spark, raw, tmp_path):
    root, labels = raw
    summary = etl.convert_all(spark, root, tmp_path, workers=2, merge_gap=10).set_index("attack")

    assert summary.loc["arp_mitm", "rows"] == 500
    assert summary.loc["arp_mitm", "first_attack_idx"] == 300
    assert summary.loc["mirai_botnet", "episodes"] == 2
    assert summary["features_as_expected"].all()
    assert (tmp_path / "summary.json").exists()

    out = spark.read.parquet(str(tmp_path / "features")).toPandas()
    assert len(out.columns) == N_FEATURES + 3  # idx, label, attack
    for name, expected in labels.items():
        part = out[out["attack"] == name].sort_values("idx")
        assert part["idx"].tolist() == list(range(len(expected)))
        # Row sequence is correct: f0 holds the source row position.
        assert part["f0"].tolist() == list(range(len(expected)))
        assert part["label"].tolist() == expected.tolist()


def test_row_count_mismatch_fails(spark, tmp_path):
    root = tmp_path / "raw"
    _write(root / "Fuzzing" / "Fuzzing_dataset.csv", _csv(_features(50, 3)))
    _write(root / "Fuzzing" / "Fuzzing_labels.csv", "\n".join(["0"] * 40) + "\n")
    with pytest.raises(RuntimeError, match="50 feature rows but 40 labels"):
        etl.convert_all(spark, root, tmp_path / "out")
