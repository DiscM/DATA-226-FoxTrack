import sys, json, numpy as np, pandas as pd
scores_dir, repo = sys.argv[1], sys.argv[2]
pd.set_option("display.width", 250)
res = pd.concat([pd.read_csv(f"{repo}/results/baseline_pca.csv"), pd.read_csv(f"{repo}/results/supervised_leave_one_out.csv")])
res = res[(res.target_fpr == 1e-3) & (res.k == 1) & (res.n == 1)].set_index(["attack", "model"])
attacks = sorted(res.index.get_level_values(0).unique())
files = {"pca_baseline": "baseline/{a}.parquet", "gbt": "supervised/gbt_{a}.parquet"}
for model, pat in files.items():
    print(f"\n===== {model}: alert rate (%) at the 0.001-target threshold")
    print("pre-onset benign test range in 10 equal parts (oldest first) | then post-onset windows of packets: 0-1k, 1k-10k, 10k-50k, rest")
    for a in attacks:
        d = pd.read_parquet(f"{scores_dir}/" + pat.format(a=a)).sort_values("idx")
        thr = res.loc[(a, model), "threshold"]; onset = int(res.loc[(a, model), "benign_packets_before_attack"])
        al = (d["score"].to_numpy() > thr); lab = d["label"].to_numpy()
        assert lab[:onset].sum() == 0 and lab[onset] == 1, a
        pre = [al[s].mean() * 100 for s in np.array_split(np.arange(onset), 10)]
        post = [al[onset + lo: onset + hi].mean() * 100 for lo, hi in ((0, 1000), (1000, 10000), (10000, 50000), (50000, len(al) - onset))]
        mal = [lab[onset + lo: onset + hi].mean() * 100 for lo, hi in ((0, 1000), (1000, 10000), (10000, 50000))]
        print(f"{a:18s}", " ".join(f"{x:5.1f}" for x in pre), "|", " ".join(f"{x:5.1f}" for x in post), "| malicious % in windows:", " ".join(f"{x:4.0f}" for x in mal))
