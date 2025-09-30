"""DAG de Airflow para tunear, entrenar ay registrar el modelo"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from airflow.decorators import dag, task

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.features.preprocessing import PreprocessingArtifacts
    from src.models.train import TrainingResult
    from src.models.tuning import CandidateModel

logger = logging.getLogger(__name__)


def _candidate_to_dict(candidate: "CandidateModel"):
    return {"name": candidate.name, "params": candidate.params, "metrics": candidate.metrics}


def _training_result_to_dict(result: "TrainingResult"):
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
    dag_id="dag_train_register",
    schedule="@monthly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"owner": "mlops"},
    tags=["training", "registry"],
)
def training_pipeline():
    """Proceso mensual para preparar datos, tunear modelo y re-entrenarlo"""

    @task
    def prepare_data():
        from src.features.preprocessing import load_default_preprocessing

        artifacts = load_default_preprocessing(sample_rows=2000)
        logger.info("Preparando los artefactos pre-procesados %s", artifacts.local_path)
        return {
            "pipeline_path": str(artifacts.local_path),
            "schema_features": artifacts.schema.feature_columns,
        }

    @task
    def hyperparameter_tuning(_: Dict[str, str]):
        from src.models.tuning import run_hyperparameter_search

        candidates, schema, _ = run_hyperparameter_search()
        logger.info("Modelo candidato %d evaluado", len(candidates))
        return {
            "candidates": [_candidate_to_dict(candidate) for candidate in candidates],
            "feature_count": len(schema.feature_columns),
        }

    @task
    def train_final_model(_: Dict[str, object]):
        from src.models.train import train_and_register_model

        result = train_and_register_model()
        logger.info("Entrenamiento completado con id %s", result.run_id)
        return _training_result_to_dict(result)

    @task
    def register_and_publish(training_info: Dict[str, object]):
        logger.info("Resumen del entrenamiento: %s", json.dumps(training_info, indent=2))

    prep = prepare_data()
    tuning = hyperparameter_tuning(prep)
    result = train_final_model(tuning)
    register_and_publish(result)


training_dag = training_pipeline()
