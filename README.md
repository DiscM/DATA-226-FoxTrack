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

## Layout

```
notebooks/            Kaggle notebooks
src/foxtrack/etl.py       CSV to Parquet with Spark
src/foxtrack/metrics.py   detection delay, alert logic, false alarms
tests/
```

## Metrics

- **Detection delay:** packets from the start of an attack episode to the first alert.
- **False alarms:** alerts on benign packets, for each 100,000 benign packets.
- **Alert logic:** an alert occurs when k of the last n scores are more than a threshold.

## Tests

```bash
pip install -r requirements.txt
```

```bash
pytest
```

The ETL tests need Java (17 or later).
