"""Segment-level error breakdowns."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_settings
from src.io_clients.minio_client import MinioClient
from src.io_clients.mlflow_client import MLflowClientWrapper

sns.set_theme(style="whitegrid")


@dataclass
class SegmentReport:
    summary: pd.DataFrame
    figure_path: Path
    minio_uri: Optional[str]


def build_segment_report(scored_df: pd.DataFrame, target_column: str, proba_column: str, segment_column: Optional[str] = None, n_segments: int = 10) -> SegmentReport:
    settings = get_settings()
    work_dir = settings.paths.reports_path() / "segments"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Dos modos: agrupando con columna o por deciles
    if segment_column and segment_column in scored_df.columns:
        grouped = scored_df.groupby(segment_column)
        summary = grouped.apply(
            lambda g: pd.Series(
                {
                    "count": len(g),
                    "avg_proba": g[proba_column].mean(),
                    "actual_rate": g[target_column].mean(),
                    "lift": g[target_column].mean() / scored_df[target_column].mean(),
                }
            )
        ).reset_index()
    else:
        scored_df = scored_df.copy()
        scored_df["segment"] = pd.qcut(scored_df[proba_column], q=n_segments, duplicates="drop")
        summary = scored_df.groupby("segment").agg(
            count=(proba_column, "count"),
            avg_proba=(proba_column, "mean"),
            actual_rate=(target_column, "mean"),
        ).reset_index()
        summary["lift"] = summary["actual_rate"] / scored_df[target_column].mean()
        segment_column = "segment"

    figure_path = work_dir / f"errors_by_segment_{segment_column}.png"
    ax = sns.barplot(data=summary, x=segment_column, y="actual_rate")
    ax.set_title(f"Error rate by {segment_column}")
    ax.set_ylabel("Actual Positive Rate")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    mlflow_client = MLflowClientWrapper()
    with mlflow_client.start_run(run_name="segment_metrics", tags={"segment_column": segment_column or "auto"}):
        mlflow_client.log_artifact(figure_path, artifact_path="segments")
        mlflow_client.log_dict(summary.to_dict(orient="records"), "segments/summary.json")

    minio_client = MinioClient.from_settings()
    minio_uri = minio_client.upload_file(
        figure_path,
        bucket=settings.minio.bucket_reports,
        object_name=f"segments/{figure_path.name}",
    )

    return SegmentReport(summary=summary, figure_path=figure_path, minio_uri=minio_uri)


__all__ = ["build_segment_report", "SegmentReport"]
