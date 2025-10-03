"""DAG de Airflow para clasificar nuevos datos"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.config import get_settings

logger = logging.getLogger(__name__)


def _serialize_inference(artifacts):
    return {
        "run_id": artifacts.run_id,
        "predictions_path": str(artifacts.predictions_path),
        "minio_uri": artifacts.minio_uri,
        "psi_global": artifacts.psi_global,
        "should_trigger_retrain": artifacts.should_trigger_retrain,
        "rows_processed": artifacts.rows_processed,
        "metrics": artifacts.metrics,
    }


@dag(
    dag_id="dag_predict_new_data",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"owner": "mlops"},
    tags=["inference", "batch"],
)
def prediction_pipeline():

    @task
    def load_new_data():
        settings = get_settings()
        new_data_path = settings.paths.new_data_path()
        if not new_data_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de nuevos datos en {new_data_path}")

        df = pd.read_csv(new_data_path, sep=";", decimal=",", encoding="utf-8")
        logger.info("Cargado nuevo dataset desde %s con %d filas", new_data_path, len(df))
        return {
            "new_data_path": str(new_data_path),
            "rows_loaded": len(df),
            "loaded_at": datetime.utcnow().isoformat(),
        }

    @task
    def score_batch(ingestion_info: dict):
        from src.models.infer import run_batch_inference

        artifacts = run_batch_inference()
        logger.info("Inferencia completada con id %s", artifacts.run_id)
        payload = _serialize_inference(artifacts)
        payload.update(ingestion_info)
        return payload

    @task
    def publish_results(info: dict):
        clean_info = {k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()}
        logger.info("Resumen de inferencia: %s", json.dumps(clean_info, indent=2))
        return info

    @task.branch
    def evaluate_retrain_need(info: dict):
        trigger = info.get("should_trigger_retrain", False)
        logger.info("Flag de reentrenamiento: %s", trigger)    
        return "skip_retrain" if trigger else "trigger_retrain"

    ingestion = load_new_data()
    scored = score_batch(ingestion)
    publish = publish_results(scored)
    decision = evaluate_retrain_need(publish)

    trigger = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="dag_model_training_deployment",
        reset_dag_run=True,
        wait_for_completion=False,
    )
    skip = EmptyOperator(task_id="skip_retrain")

    decision >> [trigger, skip]


prediction_dag = prediction_pipeline()
