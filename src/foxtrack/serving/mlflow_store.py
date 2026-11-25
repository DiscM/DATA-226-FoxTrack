"""Load the trained models from an MLflow record.

Notebook 02 records each training run in MLflow: the parameters, the metrics,
and the trained model as a file of the run. This module selects the models
for the service from that record. MLflow is thus the link between training
and deployment: the service states the MLflow run of each model that it uses.

The record is a folder with the two parts that the Kaggle run writes:

    <folder>/mlflow.db    the MLflow database (runs, parameters, metrics)
    <folder>/mlruns/      the files of each run (the trained models)

MLflow stores the location of the run files as it was on the training
machine (for example ``/kaggle/working/mlruns/1/<run>/artifacts``). This
module translates that location to the local folder.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from foxtrack.serving.model import PcaModel

EXPERIMENT = "baseline_pca_onset"


def _local_artifact_dir(mlflow_dir: Path, artifact_uri: str) -> Path:
    """Translate the stored location of the run files to the local folder."""
    path = artifact_uri.replace("file://", "")
    marker = "mlruns/"
    if marker not in path:
        raise ValueError(f"Cannot find 'mlruns/' in the run location '{artifact_uri}'")
    return mlflow_dir / "mlruns" / path.split(marker, 1)[1]


def load_models(mlflow_dir: str | Path, experiment: str = EXPERIMENT) -> dict[str, PcaModel]:
    """The newest completed run of each attack in the MLflow experiment.

    Each model gets the attribute ``source``, with the MLflow run that made it.
    """
    from mlflow.tracking import MlflowClient

    mlflow_dir = Path(mlflow_dir)
    database = mlflow_dir / "mlflow.db"
    if not database.exists():
        raise FileNotFoundError(f"No MLflow database at {database}")

    # MLflow can change a database when it opens it. Use a copy, so that the record stays as it is.
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "mlflow.db"
        shutil.copy(database, copy)
        client = MlflowClient(tracking_uri=f"sqlite:///{copy}")
        found = client.get_experiment_by_name(experiment)
        if found is None:
            raise ValueError(f"The MLflow record has no experiment '{experiment}'")
        runs = client.search_runs(
            [found.experiment_id], filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"], max_results=1000,
        )

    models: dict[str, PcaModel] = {}
    for run in runs:
        attack = run.data.params.get("attack")
        if not attack or attack in models:
            continue  # no model in this run, or a newer run of this attack is already selected
        model_file = _local_artifact_dir(mlflow_dir, run.info.artifact_uri) / "model" / f"{attack}.json"
        if not model_file.exists():
            raise FileNotFoundError(
                f"MLflow run {run.info.run_id} ({attack}) has no model file at {model_file}. "
                "Copy the folder 'mlruns' of the training run next to 'mlflow.db'."
            )
        model = PcaModel.load(model_file)
        model.source = {
            "type": "mlflow",
            "experiment": experiment,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "run_start_ms": run.info.start_time,
            "params": dict(run.data.params),
        }
        models[attack] = model
    if not models:
        raise ValueError(f"The experiment '{experiment}' has no completed run with a model")
    return dict(sorted(models.items()))
