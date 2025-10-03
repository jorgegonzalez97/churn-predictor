"""Airflow DAG para correr el PSI y los DQ"""
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
    from src.features.psi import DataQualityArtifacts

logger = logging.getLogger(__name__)


def _serialize_artifacts(artifacts: "DataQualityArtifacts") -> Dict[str, str | float | None]:
    return {
        "run_id": artifacts.run_id,
        "psi_table_path": str(artifacts.psi_table_path),
        "quality_summary_path": str(artifacts.quality_summary_path),
        "report_html_path": str(artifacts.report_html_path),
        "report_pdf_path": str(artifacts.report_pdf_path) if artifacts.report_pdf_path else None,
        "minio": artifacts.minio_uris,
    }


@dag(
    dag_id="dag_data_quality_psi",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"owner": "mlops"},
    tags=["quality", "psi"],
)
def data_quality_pipeline():
    """Pipeline diaria para asegurar el abseline y gerarar PSI"""

    @task
    def ensure_baseline() -> Dict[str, int]:
        from src.features.psi import extract_baseline_if_missing

        baseline_df = extract_baseline_if_missing()
        logger.info("Baseline listo con %d filas", len(baseline_df))
        return {"rows": len(baseline_df)}

    @task
    def compute_reports(_: Dict[str, int]):
        from src.features.psi import compute_quality_and_psi

        artifacts = compute_quality_and_psi()
        logger.info("PSI calculado con id %s", artifacts.run_id)
        return _serialize_artifacts(artifacts)

    @task
    def publish_summary(artifacts_dict):
        logger.info("Data Quality: %s", json.dumps(artifacts_dict, indent=2))
        return artifacts_dict["run_id"]

    baseline_info = ensure_baseline()
    artifacts_info = compute_reports(baseline_info)
    publish_summary(artifacts_info)


data_quality_dag = data_quality_pipeline()
