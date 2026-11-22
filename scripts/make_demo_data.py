"""Build the demo packets for the dashboard from downloaded Kitsune files.

Usage:
    python scripts/make_demo_data.py <kitsune_dir>

<kitsune_dir> has one folder for each recording, as on Kaggle, for example:
    <kitsune_dir>/Mirai Botnet/Mirai_dataset.csv
    <kitsune_dir>/Mirai Botnet/mirai_labels.csv

The script builds the demo file for each recording that it finds, and writes
it to demo_data/. A recording that is not in <kitsune_dir> is not built.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from foxtrack.demo import build_all  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    result = build_all(sys.argv[1], REPO / "models" / "pca_baseline", REPO / "demo_data")
    print("Built:", ", ".join(result["built"]) or "none")
    if result["missing"]:
        print("Not in the folder:", ", ".join(result["missing"]))
