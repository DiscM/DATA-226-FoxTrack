# FoxTrack

Early attack detection on large-scale network traffic. DATA 226 group project.

The project uses the [Kitsune Network Attack Dataset](https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune)
(nine attack captures, 115 features for each packet) and measures how quickly a model
detects the start of an attack, and how many false alarms it gives.

## Where the work runs

| Work | Location |
|---|---|
| Spark ETL, EDA, model training | Kaggle Notebook (the dataset is already there) |
| FastAPI service, dashboard | Laptop (needs only the saved model) |

## Start

1. Push this repository to GitHub.
2. In Kaggle, select **Create > Notebook > File > Import Notebook** and import
   `notebooks/01_kaggle_setup_etl.ipynb`.
3. Obey the instructions in the first cell of the notebook, then run all cells.

The notebook writes Parquet files to `/kaggle/working/kitsune_parquet`:

```
features/attack=<name>/*.parquet   idx, label, f0 .. f114
summary.json                       row counts and attack episodes
```

`idx` is the packet position in the capture. It is the time axis of the project.

4. In notebook 01, select **Save Version > Save & Run All (Commit)** to keep the output.
5. Import `notebooks/02_baseline_anomaly.ipynb`, add the output of notebook 01 as input,
   and run all cells. It trains the baseline detector and reports the detection delay.
6. Import `notebooks/03_supervised_leave_one_out.ipynb` with the same input. It trains the
   supervised models on eight captures and tests each model on the ninth.

7. `notebooks/04_time_confound_test.ipynb` answers this question for the supervised models:
   *Did the model learn what an attack looks like, or did it only learn that attacks come near the end of each recording?*
   Set `VARIANT` in its first code cell.

## Dataset facts (from the ETL summary)

- 21.0 million packets in nine captures, 115 features each.
- Each capture starts with a long benign period (1.3 to 2.7 million packets;
  Mirai: 121,621), then the attack starts one time.
- A supervised model with a time split in one capture thus cannot test the
  delay at the attack start. The baseline trains on benign packets only.

## Layout

```
notebooks/            Kaggle notebooks
src/foxtrack/etl.py       CSV to Parquet with Spark
src/foxtrack/split.py     train / calibration / test ranges for each capture
src/foxtrack/anomaly.py   baseline: scaler + PCA reconstruction error (Spark MLlib)
src/foxtrack/supervised.py  logistic regression and gradient-boosted trees, leave one capture out
src/foxtrack/metrics.py   detection delay, alert logic, false alarms
src/foxtrack/serving/     the trained detector and the online alert rules, in NumPy
src/foxtrack/api/         FastAPI service and the dashboard page
models/pca_baseline/      trained models, exported by notebook 02
demo_data/                packets near each attack start, for the dashboard
tests/
```

## Metrics

- **Detection delay:** packets from the start of an attack episode to the first alert.
- **False alarms:** alerts on benign packets before the attack. The primary measure is the
  count of events (a sequence of alerts is one event), with a 95% upper limit of the rate.
- **Alert rules:** a plain threshold; k of the last n scores more than the threshold; and a
  moving mean of the last 100 scores. The moving mean removes single high scores of benign
  packets, which cause most false alarms.
- **Credibility check:** a delay counts only if the detector reacts to the attack (the alert
  rate on malicious packets is at least 5 times the benign alert rate), and false alarms
  alone would give an alert that quickly in at most 5% of cases.

- **Placebo test:** the same evaluation at false attack starts in benign traffic. The
  credibility check must accept none of them.

## Tests

```bash
pip install -r requirements.txt
```

```bash
pytest
```

The ETL tests need Java (17 or later).

## Service and dashboard

The service gives scores and alerts for packets, and it shows a dashboard. It uses the trained
models in `models/pca_baseline/` (one for each recording). It needs NumPy only, not Spark.

```bash
pip install -r requirements-serve.txt
```

```bash
uvicorn foxtrack.api.app:app --app-dir src
```

Then open http://127.0.0.1:8000 for the dashboard, or http://127.0.0.1:8000/docs for the API.

The dashboard replays 10,000 recorded packets near an attack start (`demo_data/`) through the
detector. It shows the score, the alerts, the detection delay, and the false alarm events.

| Endpoint | Purpose |
|---|---|
| `GET /api/models` | The trained models and their settings |
| `POST /api/predict` | Scores and alerts for a sequence of packets (115 features each) |
| `POST /api/sessions`, `POST /api/sessions/{id}/packets` | A detector that keeps its history, for a stream of packets |
| `GET /api/demo/{model}` | Replay of the recorded packets near the attack start |
| `GET /api/results` | Results of the full recordings for one setting |

Example:

```bash
curl -s http://127.0.0.1:8000/api/demo/mirai_botnet | python -c "import sys, json; print(json.load(sys.stdin)['summary'])"
```

The service is a demonstration. It is not ready for a production network (see `results/RUNS.md`
and the limits in the project report).

## Results and figures

- `results/` has the result tables of the Kaggle runs.
- `results/RUNS.md` is the record of each run: code version, time, log, and corrections.
- `results/logs/` has a copy of the Kaggle log of each run.
- `results/mlflow/` has the MLflow record of the final baseline run: the database file and a
  CSV summary of the nine runs (parameters and metrics). To open it in the MLflow page:

```bash
mlflow ui --backend-store-uri sqlite:///results/mlflow/baseline_pca_run9.db
```

- `reports/figures/dashboard_*.png` are screenshots of the dashboard with the real models.
- `reports/figures/alert_rate_near_attack_start.png` shows the alert rate before and after
  the labelled attack start for each capture. To make it again, download the score files
  of notebooks 02 and 03, then run:

```bash
python scripts/plot_alert_rate_near_onset.py <scores_dir>
```
