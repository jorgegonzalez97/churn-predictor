"""Análisis de métricas del modelo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.config import get_settings
from src.io_clients.minio_client import MinioClient
from src.io_clients.mlflow_client import MLflowClientWrapper
from src.utils.ml_utils import evaluate_predictions
from src.utils.plotting import plot_calibration, plot_lift, plot_precision_recall, plot_roc, save_figure


@dataclass
class MonitoringReport:
    metrics: Dict[str, float]
    figures: Dict[str, Path]
    minio_uris: Dict[str, str]


def compute_post_deployment_metrics(scored_df, target_column: str, proba_column: str, period_column = None, output_dir = None) -> MonitoringReport:
    settings = get_settings()
    output_dir = output_dir or (settings.paths.reports_path() / "monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_predictions(scored_df[target_column].values, scored_df[proba_column].values)

    figures = {}
    figures["roc"] = save_figure(plot_roc(scored_df[target_column].values, scored_df[proba_column].values), output_dir / "roc.png")
    figures["pr"] = save_figure(plot_precision_recall(scored_df[target_column].values, scored_df[proba_column].values), output_dir / "pr.png")
    figures["lift"] = save_figure(plot_lift(scored_df[target_column].values, scored_df[proba_column].values), output_dir / "lift.png")
    figures["calibration"] = save_figure(plot_calibration(scored_df[target_column].values, scored_df[proba_column].values), output_dir / "calibration.png")

    mlflow_client = MLflowClientWrapper()
    with mlflow_client.start_run(run_name="monitoring_metrics"):
        mlflow_client.log_metrics(metrics)
        for name, path in figures.items():
            mlflow_client.log_artifact(path, artifact_path="monitoring")

    minio_client = MinioClient.from_settings()
    minio_uris = {}
    for name, path in figures.items():
        object_name = f"monitoring/{name}/{path.name}"
        uri = minio_client.upload_file(path, bucket=settings.minio.bucket_reports, object_name=object_name)
        minio_uris[name] = uri

    return MonitoringReport(metrics=metrics, figures=figures, minio_uris=minio_uris)


__all__ = ["compute_post_deployment_metrics", "MonitoringReport"]
