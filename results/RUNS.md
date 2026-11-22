# Record of runs

This file is the record of each run on Kaggle, the code version that it used, and what it
showed. All times are UTC. All runs used a Kaggle CPU notebook (4 cores, 30 GB RAM).

## How to verify a result

Each result has four records that agree with each other:

1. **The Kaggle notebook version.** Kaggle keeps each run with its code, its full log, and its
   output files. Open the notebook in your Kaggle account and select the version.
2. **The run log** in `results/logs/`. It is a copy of the Kaggle log. The Kaggle username in
   file paths is replaced with `<kaggle-user>`.
3. **The code version.** Each log has a line `Code version: <commit>`. That commit is in the
   git history of this repository, so the exact code of the run is available.
4. **The result table** in `results/`. Each table is the CSV text that the run printed in its
   log.

To make a result again: check out the commit, import the notebook into Kaggle, add the
dataset `ymirsky/network-attack-dataset-kitsune` (notebook 01) or the output of notebook 01
(other notebooks), and run all cells.

## Runs

| # | Kaggle notebook | Version | Code commit | Completed | Run time | Log | Result table |
|---|---|---|---|---|---|---|---|
| 1 | Notebook 01 (ETL), run by the project owner | - | 538bbc5 or later | 2026-11-21 23:02 | not recorded | not copied | `summary.json` in the Kaggle output |
| 2 | `foxtrack-02-baseline-anomaly` | 1 | e373171 | 2026-11-21 23:52 | 12 min 34 s | `02_baseline_v1.log` | replaced by run 9 (see git history of `baseline_pca.csv`) |
| 3 | `foxtrack-03-supervised-leave-one-out` | 1 | 91c856c | 2026-11-22 00:36 | 47 min 50 s | `03_supervised_v1.log` | `supervised_leave_one_out.csv` |
| 4 | `foxtrack-04-confound-clock` | 1 | e6f9ea8 | 2026-11-22, before 01:37 | 26 min 40 s | `04_confound_clock_v1.log` | `clock_auc.csv` |
| 5 | `foxtrack-04-confound-time-matched` | 1 | e6f9ea8 | 2026-11-22, before 01:37 | 26 min 56 s | `04_confound_time_matched_v1.log` | `supervised_time_matched.csv` |
| 6 | `foxtrack-04-confound-drop-clock` | 1 | e6f9ea8 | 2026-11-22 01:39 | 41 min 57 s | `04_confound_drop_clock_v1.log` | `supervised_drop_clock.csv` |
| 7 | `foxtrack-02-baseline-anomaly` | 2 | 27069f9 | 2026-11-22 01:48 | 10 min 39 s | `02_baseline_v2_moving_mean.log` | replaced by run 9 |
| 8 | `foxtrack-02-baseline-anomaly` | 3 | 6c4ff4c | 2026-11-22 02:12 | 10 min 35 s | `02_baseline_v3_events_placebo.log` | same numbers as run 9 |
| 9 | `foxtrack-02-baseline-anomaly` | 4 | fb9e2d9 | 2026-11-22 02:42 | 13 min 33 s | `02_baseline_v4_model_export.log` | `baseline_pca.csv`, `models/pca_baseline/`, `demo_data/` |

Notes:

- Run times are the `Wall time` lines of the log (the sum, if the log has two).
- The exact completion time of runs 4 and 5 was not recorded. They were complete when the
  status was read, before 01:37.
- The credibility columns in `supervised_leave_one_out.csv` were calculated after run 3 with
  `add_credibility`, from the columns that the run printed.
- `auc_by_attack.csv` was calculated after the runs, from the score files of runs 2, 3, 5,
  and 6, with `scripts/analysis/auc_all.py`.

## Service check

The service scores the demo packets with the exported models of run 9. For each of the nine
recordings, the largest relative difference from the scores of the Kaggle run is less than
1e-10, and the detection delays of the replay are equal to the delays of the full run (for
example Mirai 42 packets and OS Scan 30 packets, rule "mean of 100", target rate 0.0001).
`reports/figures/dashboard_mirai.png` is a screenshot of the dashboard with these models.

## Data that is not in this repository

- **Score for each packet** (45 Parquet files, approximately 335 MB). They are in the
  output of the Kaggle notebooks, folders `baseline/scores`, `supervised/scores`, and
  `confound/scores`. The figure and `auc_by_attack.csv` come from these files.
- **MLflow records.** Each Kaggle run has a file `mlflow.db` in its output. The record of run 9
  is in this repository: `results/mlflow/baseline_pca_run9.db`, with a CSV summary. The records
  of the other runs are only on Kaggle.
- **The Parquet data** (21.0 million packets). It is the output of notebook 01.

## Analyses outside the Kaggle runs

These ran on the downloaded score files. The scripts are in `scripts/analysis/`. Their output
was reported in the project conversation and is not saved as a file, except where stated.

| Script | Purpose | Saved output |
|---|---|---|
| `near_onset.py` | Alert rate in 10 parts of the benign period, and after the attack start | none |
| `smooth_test.py` | First test of the moving mean (see Corrections, item 2) | none |
| `confound_report.py` | Comparison of the standard trees with the two corrections | `supervised_drop_clock.csv`, `supervised_time_matched.csv`, `clock_auc.csv` |
| `auc_all.py` | ROC-AUC and PR-AUC for each attack and model | `auc_by_attack.csv` |
| `../plot_alert_rate_near_onset.py` | The figure | `reports/figures/alert_rate_near_attack_start.png` |

## Corrections

Statements that were made during the project and then corrected by a later result.

1. **Cause of the early alerts of the trees.** First hypothesis: the attack in ARP MitM and
   Active Wiretap starts before the labelled start. Runs 4 to 6 showed a different cause: the
   model learned the time in the recording. Evidence: a model trained on benign packets only
   predicts early or late with a ROC-AUC of more than 0.85 in six of nine recordings, and
   with time-matched training the early alerts stop.
2. **Moving mean and Mirai.** The first test (`smooth_test.py`) showed zero false alarms for
   Mirai, OS Scan, and SSDP Flood. That test took its threshold from benign packets
   immediately before the evaluation packets. Run 7 used the correct calibration range and
   showed false alarms for Mirai (4 events in 18,244 benign packets). The result for OS Scan
   and SSDP Flood stayed.
3. **False alarm unit.** Results before run 8 count packets with a false alarm. For the moving
   mean this count is too large, because one event continues for approximately 100 packets.
   Run 8 counts events, and the credibility check uses events.
4. **ROC-AUC of the trees for ARP MitM (0.980) and Active Wiretap (0.771).** These values of
   run 3 are an effect of the time in the recording. With time-matched training (run 5) they
   are 0.548 and 0.656.

## Problems during setup

| Problem | Cause | Correction |
|---|---|---|
| ETL stopped: "more than one partition" | Spark divided each large plain CSV into one part for each core | Commit 538bbc5 sets two Spark file settings; a test with a large CSV reproduces the fault |
| MLflow rejected the file store | The newest MLflow needs a database store | Commit e373171 uses a SQLite store |
| A header line of column numbers looked like data | Header detection tested only for text | Corrected before the first commit; a test covers it |
