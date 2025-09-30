"""DAG de Airflow para clasificar nuevos datos"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.models.infer import InferenceArtifacts

logger = logging.getLogger(__name__)


def _serialize_inference(artifacts: "InferenceArtifacts"):
    return {
        "run_id": artifacts.run_id,
        "predictions_path": str(artifacts.predictions_path),
        "minio_uri": artifacts.minio_uri,
        "psi_global": artifacts.psi_global,
        "should_trigger_retrain": artifacts.should_trigger_retrain,
    }


@dag(
    dag_id="dag_predict_new_data",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"owner": "mlops"},
    tags=["inference", "batch"],
)
def prediction_pipeline():

    @task
    def load_new_data():
        logger.info("Futura mejora: paso para implementar ingesta de datos")
        return datetime.utcnow().isoformat()

    @task
    def score_batch(_: str):
        from src.models.infer import run_batch_inference

        artifacts = run_batch_inference()
        logger.info("Inferencia completada con id %s", artifacts.run_id)
        return _serialize_inference(artifacts)

    @task
    def publish_results(info: Dict[str, object]):
        logger.info("Resumen de inferencia: %s", json.dumps(info, indent=2))
        return info

    @task.branch
    def evaluate_retrain_need(info: Dict[str, object]):
        trigger = bool(info.get("should_trigger_retrain", False))
        return "trigger_retrain" if trigger else "skip_retrain"

    ingestion = load_new_data()
    scored = score_batch(ingestion)
    publish = publish_results(scored)
    decision = evaluate_retrain_need(publish)

    trigger = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="dag_train_register",
        reset_dag_run=True,
        wait_for_completion=False,
    )
    skip = EmptyOperator(task_id="skip_retrain")

    decision >> [trigger, skip]


prediction_dag = prediction_pipeline()
