"""DAG de Airflow para entrenar el mejor modelo y publicarlo"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.models.train import TrainingResult

logger = logging.getLogger(__name__)


def _training_result_to_dict(result: "TrainingResult") -> Dict[str, object]:
    return {
        "run_id": result.run_id,
        "model_name": result.model_name,
        "metrics": result.metrics,
        "local_model_path": str(result.local_model_path),
        "minio_uri": result.minio_uri,
        "mlflow_run_id": result.mlflow_run_id,
        "mlflow_model_version": result.mlflow_model_version,
    }


@dag(
    dag_id="dag_model_training_deployment",
    schedule="@monthly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"owner": "mlops"},
    tags=["training", "deployment"],
)
def model_training_deployment_pipeline():
    """Pipeline mensual para entrenar el mejor modelo y publicarlo."""

    @task
    def train_final_model() -> Dict[str, object]:
        from src.models.train import train_and_register_model

        result = train_and_register_model()
        logger.info("Entrenamiento completado con id %s", result.run_id)
        return _training_result_to_dict(result)

    @task
    def register_and_publish(training_info: Dict[str, object]) -> None:
        logger.info("Resumen del entrenamiento: %s", json.dumps(training_info, indent=2))

    register_and_publish(train_final_model())


model_training_deployment_dag = model_training_deployment_pipeline()
