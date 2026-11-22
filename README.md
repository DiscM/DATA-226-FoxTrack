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
tests/
```

## Metrics

- **Detection delay:** packets from the start of an attack episode to the first alert.
- **False alarms:** alerts on benign packets, for each 100,000 benign packets.
- **Alert logic:** an alert occurs when k of the last n scores are more than a threshold.
- **Credibility check:** a delay counts only if the detector reacts to the attack (the alert
  rate on malicious packets is at least 5 times the benign alert rate), and false alarms
  alone would give an alert that quickly in at most 5% of cases.

## Tests

```bash
pip install -r requirements.txt
```

```bash
pytest
```

The ETL tests need Java (17 or later).
