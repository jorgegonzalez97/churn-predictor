"""Predicción para nuevos datos."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.config import get_settings
from src.features.psi import compute_psi_table, extract_baseline_if_missing
from src.features.schema_inference import DataSchema, infer_schema, load_dataset
from src.io_clients.minio_client import MinioClient
from src.io_clients.mlflow_client import MLflowClientWrapper
from src.io_clients.postgres_logging import PostgresLogger
from src.utils.ml_utils import evaluate_predictions

logger = logging.getLogger(__name__)


@dataclass
class InferenceArtifacts:
    run_id: str
    predictions_path: Path
    minio_uri: Optional[str]
    psi_global: float
    should_trigger_retrain: bool
    rows_processed: int
    metrics: Dict[str, float]


MODEL_REGISTRY_NAME = "churn-risk-model"

#Cargar el modelo de Produccion desde MLFlow
def _load_model_from_registry(model_name: str) -> Optional[object]:
    try:
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/Production")
        logger.info("Modelo cargado del registro de MLflow")
        return model
    except Exception as exc: 
        logger.warning("Algo falló cargando el modelo del registro de MLFlow: %s", exc)
        return None

# Cargar el último modelo de la carpeta de modelos 
def _load_latest_local_model() -> object:
    settings = get_settings()
    models_dir = settings.paths.models_path()
    if not models_dir.exists():
        raise FileNotFoundError("No se ha encontrado el directorio local de modelos")
    candidates = sorted(models_dir.glob("churn_model_*.joblib"))
    if not candidates:
        raise FileNotFoundError("No se han encontrado artefactos entrenados")
    latest = candidates[-1]
    logger.info("Volviendo a la versión local %s", latest)
    packed = joblib.load(latest)
    return packed["pipeline"]

# primero prueba el registro de MLflow, si no hay suerte, usa el último modelo local.
def load_scoring_pipeline():
    model = _load_model_from_registry(MODEL_REGISTRY_NAME)
    if model is not None:
        return model
    return _load_latest_local_model()


def _prepare_new_data(schema: Optional[DataSchema] = None) -> pd.DataFrame:
    settings = get_settings()
    new_data_path = settings.paths.new_data_path()
    if not new_data_path.exists():
        logger.warning("El nuevo archivo de datos no se encuentra en %s; generamos muestra sintética", new_data_path)
        df = load_dataset().sample(n=500, random_state=settings.pipeline.random_state)
        df.to_csv(new_data_path, index=False)
        return df
    if new_data_path.suffix.lower() == ".csv":
        return pd.read_csv(new_data_path, sep=";", decimal=",", encoding="utf-8")
    if new_data_path.suffix.lower() == ".parquet":
        return pd.read_parquet(new_data_path)
    raise ValueError(f"Formato no soportado: {new_data_path.suffix}")


def run_batch_inference() -> InferenceArtifacts:
    settings = get_settings()
    schema = infer_schema(load_dataset())
    pipeline = load_scoring_pipeline()

    new_data = _prepare_new_data(schema)
    features = new_data[schema.feature_columns]
    predictions = pipeline.predict(features)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)[:, 1]
    else:
        probabilities = np.clip(pipeline.predict(features), 0, 1)

    result_df = new_data.copy()
    result_df["proba_churn"] = probabilities
    result_df["prediction"] = predictions

    run_id = uuid.uuid4().hex[:12]
    preds_dir = settings.paths.data_dir
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"predictions_{run_id}.csv"
    result_df.to_csv(preds_path, index=False)

    minio_client = MinioClient.from_settings()
    minio_uri = None
    try:
        object_name = f"predictions/{run_id}/predictions.csv"
        minio_uri = minio_client.upload_file(preds_path, bucket=settings.minio.bucket_predictions, object_name=object_name)
    except Exception as exc: 
        logger.warning("La subida de predicciones a MinIO ha fallado: %s", exc)

    baseline_df = extract_baseline_if_missing()
    psi_df = compute_psi_table(baseline_df, new_data, schema)
    psi_global = float(psi_df["psi"].mean()) if not psi_df.empty else 0.0
    should_trigger_retrain = psi_global >= settings.pipeline.psi_threshold
    rows_processed = len(new_data)

    if schema.target in new_data.columns:
        metrics = evaluate_predictions(new_data[schema.target].values, probabilities)
        should_trigger_retrain = should_trigger_retrain or (metrics.get("pr_auc", 1.0) < settings.pipeline.pr_auc_shadow_threshold)
    else:
        metrics = {}

    mlflow_client = MLflowClientWrapper()
    with mlflow_client.start_run(run_name="batch_inference", tags={"run_id": run_id}) as mlflow_run:
        mlflow_client.log_metrics({"psi_global": psi_global})
        if metrics:
            mlflow_client.log_metrics(metrics)
        mlflow_client.log_metrics({"should_trigger_retrain": bool(should_trigger_retrain)})
        mlflow_client.set_tags(
            {
                "retrain_candidate": str(should_trigger_retrain).lower(),
                "prediction_rows": str(rows_processed),
            }
        )
        mlflow_client.log_artifact(preds_path, artifact_path="predictions")
        mlflow_client.log_dict(psi_df.to_dict(orient="records"), "monitoring/psi.json")
        mlflow_client.log_dict(
            {
                "rows_processed": rows_processed,
                "psi_global": psi_global,
                "should_trigger_retrain": should_trigger_retrain,
                "mlflow_run_id": mlflow_run.run_id,
            },
            "monitoring/summary.json",
        )

    pg_logger = PostgresLogger.from_settings()
    try:
        pg_logger.ensure_tables()
        pg_logger.log_scoring_run(
            run_id=run_id,
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            rows_processed=rows_processed,
            latency_ms=0,
            error_rate=float(1 - metrics.get("accuracy", 1.0)) if metrics else 0.0,
            notes=json.dumps(
                {
                    "psi_global": psi_global,
                    "metrics": metrics,
                    "should_trigger_retrain": should_trigger_retrain,
                    "rows_processed": rows_processed,
                }
            ),
        )
    except Exception as exc:
        logger.warning("El registro en Postgres ha fallado: %s", exc)

    return InferenceArtifacts(
        run_id=run_id,
        predictions_path=preds_path,
        minio_uri=minio_uri,
        psi_global=psi_global,
        should_trigger_retrain=should_trigger_retrain,
        rows_processed=rows_processed,
        metrics=metrics,
    )


__all__ = ["run_batch_inference", "InferenceArtifacts"]
