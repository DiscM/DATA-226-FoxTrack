import sys, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
scores, out = sys.argv[1], sys.argv[2]
pd.set_option("display.width", 250)
attacks = ["arp_mitm", "active_wiretap", "fuzzing", "mirai_botnet", "os_scan", "ssdp_flood", "ssl_renegotiation", "syn_dos", "video_injection"]
models = {"pca_baseline": "baseline/{a}.parquet", "logreg": "supervised/logreg_{a}.parquet", "gbt_standard": "supervised/gbt_{a}.parquet",
          "gbt_drop_clock": "confound/gbt_drop_clock_{a}.parquet", "gbt_time_matched": "confound/gbt_time_matched_{a}.parquet"}
rows = []
for a in attacks:
    for m, pat in models.items():
        d = pd.read_parquet(f"{scores}/" + pat.format(a=a))
        rows.append({"attack": a, "model": m, "roc_auc": roc_auc_score(d.label, d.score), "pr_auc": average_precision_score(d.label, d.score), "malicious_fraction": d.label.mean()})
r = pd.DataFrame(rows); r.to_csv(out, index=False)
print(r.pivot(index="attack", columns="model", values="roc_auc")[list(models)].round(3).to_string())
print(r.groupby("model")["roc_auc"].agg(["mean", "median", "min"]).loc[list(models)].round(3).to_string())
print(r.groupby("attack")["malicious_fraction"].first().round(3).to_string())
