import sys, numpy as np, pandas as pd
out04, scores, repo = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo + "/src")
from foxtrack.metrics import add_credibility
pd.set_option("display.width", 250)
def from_log(path):
    lines = open(path).read().splitlines()
    i = max(k for k, l in enumerate(lines) if l.startswith("attack,model,target_fpr"))
    rows = [lines[i]] + [l for l in lines[i + 1:] if l.count(",") == lines[i].count(",")]
    import io; return pd.read_csv(io.StringIO("\n".join(rows)))
drop = from_log(f"{out04}/drop-clock.log"); tm = from_log(f"{out04}/time-matched.log")
drop.to_csv(f"{repo}/results/supervised_drop_clock.csv", index=False); tm.to_csv(f"{repo}/results/supervised_time_matched.csv", index=False)
orig = pd.read_csv(f"{repo}/results/supervised_leave_one_out.csv"); orig = orig[orig.model == "gbt"].assign(model="gbt_standard")
allr = add_credibility(pd.concat([orig, drop, tm], ignore_index=True))
at = allr[(allr.target_fpr == 1e-3) & (allr.k == 1) & (allr.n == 1)]
print("ROC-AUC"); print(allr.groupby(["attack", "model"])["roc_auc"].first().unstack().round(3).to_string())
print("\nFalse alarms per 100k before attack (target 100)"); print(at.pivot(index="attack", columns="model", values="false_alarms_per_100k").round(0).to_string())
print("\nPacket recall"); print(at.pivot(index="attack", columns="model", values="packet_recall").round(3).to_string())
print("\nAlert lift"); print(at.pivot(index="attack", columns="model", values="alert_lift").round(1).to_string())
print("\nDelay (* credible)"); print(at.assign(d=at.delay_packets.map(lambda x: "-" if pd.isna(x) else str(int(x))) + at.delay_credible.map({True: "*", False: ""})).pivot(index="attack", columns="model", values="d").to_string())
print("\nCredible delays, any setting, FA <= 100 per 100k:")
ok = allr[allr.delay_credible & (allr.false_alarms_per_100k <= 100)].sort_values("delay_packets").groupby(["attack", "model"]).first()[["delay_packets", "target_fpr", "k", "n", "false_alarms_per_100k", "packet_recall"]]
print(ok.round(3).to_string())
# clock AUC
import re
rows = [re.match(r"(\w+): clock ROC-AUC ([\d.]+)", l).groups() for l in open(f"{out04}/clock.log").read().splitlines() if "clock ROC-AUC" in l]
pd.DataFrame(rows, columns=["attack", "clock_roc_auc"]).to_csv(f"{repo}/results/clock_auc.csv", index=False)
