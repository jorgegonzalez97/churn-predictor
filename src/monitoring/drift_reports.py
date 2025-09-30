"""Report de Drift de datos"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from typing import TYPE_CHECKING

from src.config import get_settings
from src.features.psi import compute_psi_table
from src.features.schema_inference import DataSchema, infer_schema, load_dataset
from src.io_clients.minio_client import MinioClient

if TYPE_CHECKING:
    from evidently import Report

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    run_id: str
    html_path: Path
    psi_path: Path
    minio_uris: dict[str, str]


def generate_drift_report(current_df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None, schema: Optional[DataSchema] = None) -> DriftReport:
    settings = get_settings()
    schema = schema or infer_schema(load_dataset())
    reference_df = reference_df if reference_df is not None else load_dataset()

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[schema.feature_columns + [schema.target]],
        current_data=current_df[schema.feature_columns + [schema.target]],
    )

    run_id = uuid.uuid4().hex[:12]
    output_dir = settings.paths.reports_path() / "drift"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"drift_report_{run_id}.html"
    report.save_html(html_path)

    psi_df = compute_psi_table(reference_df, current_df, schema)
    psi_path = output_dir / f"psi_{run_id}.csv"
    psi_df.to_csv(psi_path, index=False)

    minio_client = MinioClient.from_settings()
    minio_uris = {}
    for path, key in [(html_path, "html"), (psi_path, "psi")]:
        try:
            object_name = f"monitoring/{key}/{run_id}/{path.name}"
            uri = minio_client.upload_file(path, bucket=settings.minio.bucket_reports, object_name=object_name)
            minio_uris[key] = uri
        except Exception as exc:
            logger.warning("La subidad de %s a MinIO: %s", path, exc)

    return DriftReport(run_id=run_id, html_path=html_path, psi_path=psi_path, minio_uris=minio_uris)


__all__ = ["generate_drift_report", "DriftReport"]
