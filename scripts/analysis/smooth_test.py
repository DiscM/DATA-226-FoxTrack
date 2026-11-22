"""Offline test: does a moving mean of the score decrease false alarms?
Threshold: from the first half of the benign test packets. Evaluation: second half + attack."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2] + "/src")
from foxtrack.metrics import credibility
scores_dir, repo = sys.argv[1], sys.argv[2]
pd.set_option("display.width", 250)
res = pd.read_csv(f"{repo}/results/baseline_pca.csv")
onsets = res.groupby("attack")["benign_packets_before_attack"].first()
rows = []
for a, onset in onsets.items():
    d = pd.read_parquet(f"{scores_dir}/baseline/{a}.parquet").sort_values("idx")
    raw = np.log1p(d["score"].to_numpy()); lab = d["label"].to_numpy()
    half = onset // 2
    for w in (1, 20, 100, 500):
        s = pd.Series(raw).rolling(w, min_periods=1).mean().to_numpy()   # uses only past packets
        for fpr in (1e-3, 1e-4):
            thr = np.quantile(s[:half], 1 - fpr)
            al = s > thr
            fa = int(al[half:onset].sum()); nb = onset - half
            after = np.flatnonzero(al[onset:]); delay = int(after[0]) if after.size else None
            rec = float(al[onset:][lab[onset:] == 1].mean())
            c = credibility(delay, fa, nb, rec)
            rows.append({"attack": a, "window": w, "target_fpr": fpr, "fa_per_100k": fa / nb * 1e5, "delay": delay, "recall": rec, "credible": c["delay_credible"]})
r = pd.DataFrame(rows)
for fpr in (1e-3, 1e-4):
    at = r[r.target_fpr == fpr]
    print(f"\n=== target {fpr}: false alarms per 100k, by window"); print(at.pivot(index="attack", columns="window", values="fa_per_100k").round(1).to_string())
    print("delay (packets); * = credible"); print(at.assign(d=at.delay.map(lambda x: "-" if pd.isna(x) else str(int(x))) + at.credible.map({True: "*", False: ""})).pivot(index="attack", columns="window", values="d").to_string())
    print("recall"); print(at.pivot(index="attack", columns="window", values="recall").round(3).to_string())
