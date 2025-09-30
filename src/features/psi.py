"""Population Stability Index (PSI) and data quality routines."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from src.config import get_settings
from src.features.preprocessing import fit_preprocessing_pipeline
from src.features.schema_inference import DataSchema, infer_schema, load_dataset
from src.io_clients.minio_client import MinioClient
from src.io_clients.mlflow_client import MLflowClientWrapper

logger = logging.getLogger(__name__)
# Evitamos divisiones y logs nulos
EPSILON = 1e-6


@dataclass
class DataQualityArtifacts:
    run_id: str
    psi_table_path: Path
    quality_summary_path: Path
    report_html_path: Optional[Path]
    report_pdf_path: Optional[Path]
    minio_uris: Dict[str, str]


def extract_baseline_if_missing(current_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    settings = get_settings()
    baseline_path = settings.paths.baseline_parquet_path()
    if baseline_path.exists():
        logger.info("Loading existing baseline snapshot from %s", baseline_path)
        return pd.read_parquet(baseline_path)

    df = current_df if current_df is not None else load_dataset()
    baseline = df.sample(frac=settings.pipeline.baseline_share, random_state=settings.pipeline.random_state)
    baseline.to_parquet(baseline_path, index=False)
    logger.info("Persisted new baseline snapshot to %s", baseline_path)
    return baseline


def _categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna().unique()).union(set(current.dropna().unique())))
    ref_dist = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_dist = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    ref_dist = ref_dist.replace(0, EPSILON)
    cur_dist = cur_dist.replace(0, EPSILON)
    psi = float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())
    return psi


def _numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return 0.0
    try:
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.unique(np.quantile(reference, quantiles))
    except ValueError:
        bin_edges = np.unique(reference.values)
    if len(bin_edges) < 2:
        bin_edges = np.array([reference.min() - EPSILON, reference.max() + EPSILON])
    
    #Extendemos el primer y último borde a infinito para asegurar que todos los valores actuales caen en algún bin, aunque estén fuera del rango de referencia (importante en drift)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)
    ref_dist = np.where(ref_hist == 0, EPSILON, ref_hist / ref_hist.sum())
    cur_dist = np.where(cur_hist == 0, EPSILON, cur_hist / cur_hist.sum())
    psi = float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())
    return psi


def compute_psi_table(reference: pd.DataFrame, current: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    rows = []
    for column in schema.feature_columns:
        if column not in reference.columns or column not in current.columns:
            continue
        if column in schema.numerical:
            psi_value = _numeric_psi(reference[column], current[column])
        else:
            psi_value = _categorical_psi(reference[column], current[column])
        rows.append({"feature": column, "psi": psi_value})
    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    return psi_df


def compute_quality_summary(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    summaries = []
    for column in schema.feature_columns + [schema.target]:
        if column not in df.columns:
            continue
        series = df[column]
        summary = {
            "feature": column,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean()),
        }
        if column in schema.numerical:
            
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_count = int(((series < lower_bound) | (series > upper_bound)).sum())

            summary.update(
                {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "unique": int(series.nunique(dropna=True)),
                    "p1": float(series.quantile(0.01)), #Info para tratamiento de Outliers
                    "p99": float(series.quantile(0.99)), #Info para tratamiento de Outliers
                    "outliers_count": outliers_count
                }
            )
        else:
            top = series.mode(dropna=True)
            value_counts = series.value_counts(dropna=True).to_dict()
            summary.update(
                {
                    "unique": int(series.nunique(dropna=True)),
                    "top": None if top.empty else str(top.iloc[0]),
                    "categories": {str(k): int(v) for k, v in value_counts.items()},
                }
            )
        summaries.append(summary)
    return pd.DataFrame(summaries)

# Importamos dentro para evitamos que los DAGs se dropeen por tiempo de carga excesivo.
def generate_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, schema: DataSchema,  run_id: str, reports_dir: Path):
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    report = Report([DataSummaryPreset(), DataDriftPreset()])
    report = report.run(current_data = current, reference_data= reference)

    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / f"data_quality_{run_id}.html"
    html_generated: Optional[Path] = html_path
    
    try:
        report.save_html(html_path)
    except Exception as exc:
        logger.error("No se pudo exportar el HTML de Evidently: %s", exc)
        html_generated = None

    pdf_path = reports_dir / f"data_quality_{run_id}.pdf"
    generated_pdf: Optional[Path] = None
    if html_generated and html_generated.exists():
        try:
            from weasyprint import HTML 

            HTML(filename=str(html_generated)).write_pdf(str(pdf_path))
            generated_pdf = pdf_path
        except Exception as exc:
            logger.warning("Falló la exportación del PDF via WeasyPrint: %s", exc)
            generated_pdf = None
    
    return html_generated, generated_pdf


def compute_quality_and_psi(current_df: Optional[pd.DataFrame] = None) -> DataQualityArtifacts:
    settings = get_settings()
    df_current = current_df if current_df is not None else load_dataset()
    df_reference = extract_baseline_if_missing(df_current)

    sample_size = settings.pipeline.dq_sample_size()
    df_current_for_report = df_current
    df_reference_for_report = df_reference
    if sample_size and len(df_current) > sample_size:
        df_current_for_report = df_current.sample(
            n=sample_size,
            random_state=settings.pipeline.random_state,
        )
        logger.info(
            "Reduciendo snapshot actual de %d a %d filas para generar reportes",
            len(df_current),
            len(df_current_for_report),
        )
    if sample_size and len(df_reference) > sample_size:
        df_reference_for_report = df_reference.sample(
            n=sample_size,
            random_state=settings.pipeline.random_state,
        )
        logger.info(
            "Reduciendo baseline de %d a %d filas para generar reportes",
            len(df_reference),
            len(df_reference_for_report),
        )

    schema = infer_schema(df_reference)
    psi_df = compute_psi_table(df_reference, df_current, schema)
    quality_df = compute_quality_summary(df_current, schema)

    run_identifier = uuid.uuid4().hex[:12]
    reports_dir = settings.paths.reports_path()
    reports_dir.mkdir(parents=True, exist_ok=True)
    psi_path = reports_dir / f"psi_{run_identifier}.csv"
    quality_path = reports_dir / f"data_quality_{run_identifier}.csv"
    psi_df.to_csv(psi_path, index=False)
    quality_df.to_csv(quality_path, index=False)

    html_path, pdf_path = generate_evidently_report(
        df_reference_for_report,
        df_current_for_report,
        schema,
        run_identifier,
        reports_dir,
    )

    # Validación de preprocessing artifact
    try:
        fit_preprocessing_pipeline(df_reference, schema)
    except Exception as exc: 
        logger.warning("No se consiguió compilar la pipeline de processing durante la fase de DQ: %s", exc)

    minio_client = MinioClient.from_settings()
    minio_uris = {}
    artifacts_to_upload: list[tuple[Path, str, str]] = [
        (psi_path, settings.minio.bucket_reports, "psi"),
        (quality_path, settings.minio.bucket_reports, "quality"),
    ]
    if html_path:
        artifacts_to_upload.append((html_path, settings.minio.bucket_reports, "html"))

    for path, bucket, folder in artifacts_to_upload:
        try:
            object_name = f"reports/{folder}/{run_identifier}/{path.name}"
            uri = minio_client.upload_file(path, bucket=bucket, object_name=object_name)
            minio_uris[path.name] = uri
        except Exception as exc:
            logger.warning("La carga de datos a MinIO ha fallado por %s: %s", path, exc)
    if pdf_path:
        try:
            object_name = f"reports/pdf/{run_identifier}/{pdf_path.name}"
            uri = minio_client.upload_file(pdf_path, bucket=settings.minio.bucket_reports, object_name=object_name)
            minio_uris[pdf_path.name] = uri
        except Exception as exc:
            logger.warning("La carga del pdf a MinIO ha fallado por %s: %s", pdf_path, exc)

    mlflow_client = MLflowClientWrapper()
    with mlflow_client.start_run(run_name="data_quality_psi") as mlflow_run:
        mlflow_client.log_params({
            "run_id": run_identifier,
            "reference_rows": len(df_reference),
            "current_rows": len(df_current),
        })
        mlflow_client.log_metrics({f"psi_{row['feature']}": float(row["psi"]) for _, row in psi_df.iterrows()})
        mlflow_client.log_dict(psi_df.to_dict(orient="records"), "reports/psi.json")
        mlflow_client.log_dict(quality_df.to_dict(orient="records"), "reports/data_quality.json")
        if html_path:
            try:
                mlflow_client.log_artifact(html_path, artifact_path="reports")
            except FileNotFoundError:
                logger.warning("Reporte HTML no encontrado al registrar en MLflow: %s", html_path)
        if pdf_path:
            try:
                mlflow_client.log_artifact(pdf_path, artifact_path="reports")
            except FileNotFoundError:
                logger.warning("Reporte PDF no encontrado al registrar en MLflow: %s", pdf_path)
        mlflow_client.set_tags({"stage": "data_quality", "generated_at": datetime.utcnow().isoformat()})

    return DataQualityArtifacts(
        run_id=run_identifier,
        psi_table_path=psi_path,
        quality_summary_path=quality_path,
        report_html_path=html_path,
        report_pdf_path=pdf_path,
        minio_uris=minio_uris,
    )


__all__ = [
    "DataQualityArtifacts",
    "compute_quality_and_psi",
    "extract_baseline_if_missing",
]
