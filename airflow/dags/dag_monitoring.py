"""DAG de Airflow para kmonitorear drift y performance del modello desplegado"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

PREDICTIONS_GLOB = "predictions_*.csv"


def _find_latest_predictions(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob(PREDICTIONS_GLOB))
    if not candidates:
        raise AirflowSkipException("No hay predicciones que monitorear")
    return candidates[-1]


@dag(
    dag_id="dag_monitoring",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"owner": "mlops"},
    tags=["monitoring"],
)
def monitoring_pipeline():
    """Monitoreo semanal de drift y performance"""

    @task
    def load_scored_dataset() -> Dict[str, str]:
        from src.config import get_settings

        settings = get_settings()
        latest_path = _find_latest_predictions(settings.paths.data_dir)
        logger.info("Cargando predicciones desde %s", latest_path)
        return {"path": str(latest_path)}

    @task
    def monitor_model_performance(payload: Dict[str, str]) -> Dict[str, str]:
        import pandas as pd

        from src.features.schema_inference import infer_schema, load_dataset
        from src.monitoring.metrics import compute_post_deployment_metrics

        path = Path(payload["path"])
        df = pd.read_csv(path, sep=None, engine="python")
        schema = infer_schema(load_dataset())
        proba_column = "proba_churn"
        if schema.target not in df.columns or proba_column not in df.columns:
            logger.warning("Monitoreo de performance (faltan columnas)")
            raise AirflowSkipException("Falta el target o las probabilidades")
        report = compute_post_deployment_metrics(df, target_column=schema.target, proba_column=proba_column)
        payload.update(
            {
                "monitoring_metrics": json.dumps(report.metrics),
                "monitoring_figures": json.dumps({k: str(v) for k, v in report.figures.items()}),
                "monitoring_minio": json.dumps(report.minio_uris),
            }
        )
        return payload

    @task
    def segment_error_reports(payload: Dict[str, str]) -> None:
        import pandas as pd

        from src.features.schema_inference import infer_schema, load_dataset
        from src.monitoring.errors_by_segment import build_segment_report

        path = Path(payload["path"])
        df = pd.read_csv(path, sep=None, engine="python")
        schema = infer_schema(load_dataset())
        proba_column = "proba_churn"
        if schema.target not in df.columns or proba_column not in df.columns:
            raise AirflowSkipException("No hay suficientes columnas para reportes por segmento")
        report = build_segment_report(df, target_column=schema.target, proba_column=proba_column)
        logger.info(
            "Metricas de segmento guardadas en %s (MinIO: %s)",
            report.figure_path,
            report.minio_uri,
        )

    dataset = load_scored_dataset()
    performance_payload = monitor_model_performance(dataset)
    segment_error_reports(performance_payload)


monitoring_dag = monitoring_pipeline()
