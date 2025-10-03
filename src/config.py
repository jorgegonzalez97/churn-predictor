"""Configuracion centralizada para el pipeline"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MLRUNS_DIR = BASE_DIR / "mlruns"


def _get_env(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable '{key}' is required but missing")
    return value


def _get_env_int(key: str, default: Optional[int] = None) -> int:
    value = os.getenv(key)
    if value is None:
        if default is None:
            raise ValueError(f"Environment variable '{key}' is required but missing")
        return default
    return int(value)


def _get_env_float(key: str, default: Optional[float] = None) -> float:
    value = os.getenv(key)
    if value is None:
        if default is None:
            raise ValueError(f"Environment variable '{key}' is required but missing")
        return default
    return float(value)


def _get_env_bool(key: str, default: Optional[bool] = None) -> bool:
    value = os.getenv(key)
    if value is None:
        if default is None:
            raise ValueError(f"Environment variable '{key}' is required but missing")
        return default
    return value.lower() in {"1", "true", "yes", "si", "on"}


@dataclass(frozen=True)
class MinIOSettings:
    endpoint_url: str = _get_env("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    access_key: str = _get_env("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key: str = _get_env("AWS_SECRET_ACCESS_KEY", "minioadmin")
    secure: bool = _get_env_bool("MINIO_SECURE", False)
    bucket_artifacts: str = _get_env("S3_ARTIFACT_BUCKET", "mlflow")
    bucket_models: str = _get_env("MINIO_MODELS_BUCKET", "models")
    bucket_reports: str = _get_env("MINIO_REPORTS_BUCKET", "reports")
    bucket_predictions: str = _get_env("MINIO_PREDICTIONS_BUCKET", "predictions")

    def s3_uri(self, bucket: str, object_name: str) -> str:
        safe_object = object_name.lstrip("/")
        return f"s3://{bucket}/{safe_object}"


@dataclass(frozen=True)
class MLflowSettings:
    tracking_uri: str = _get_env("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-risk")
    registry_uri: Optional[str] = os.getenv("MLFLOW_REGISTRY_URI")
    artifact_location: str = _get_env("MLFLOW_ARTIFACT_URI", "s3://mlflow")


@dataclass(frozen=True)
class PostgresSettings:
    host: str = os.getenv("POSTGRES_HOST", "postgres")
    port: int = _get_env_int("POSTGRES_PORT", 5432)
    user: str = _get_env("POSTGRES_USER", "airflow")
    password: str = _get_env("POSTGRES_PASSWORD", "airflow")
    airflow_db: str = _get_env("POSTGRES_DB", "airflow")
    mlflow_db: str = _get_env("MLFLOW_POSTGRES_DB", "mlflow")
    monitoring_db: str = os.getenv("MONITORING_POSTGRES_DB", airflow_db)
    monitoring_schema: str = os.getenv("MONITORING_SCHEMA", "monitoring")

    def airflow_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.airflow_db}"
        )

    def mlflow_uri(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.mlflow_db}"
        )

    def monitoring_uri(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.monitoring_db}"
        )


@dataclass(frozen=True)
class PipelineSettings:
    target_column: str = os.getenv("TARGET_COLUMN", "churn")
    id_column: str = os.getenv("ID_COLUMN", "Customer_ID")
    baseline_share: float = _get_env_float("BASELINE_SHARE", 0.2)
    random_state: int = _get_env_int("RANDOM_STATE", 753)
    psi_threshold: float = _get_env_float("PSI_THRESHOLD", 0.2)
    pr_auc_shadow_threshold: float = _get_env_float("PR_AUC_SHADOW_THRESHOLD", 0.5)
    optuna_n_trials: int = _get_env_int("OPTUNA_N_TRIALS", 5)
    test_size: float = _get_env_float("MODEL_TEST_SIZE", 0.2)
    validation_size: float = _get_env_float("MODEL_VALID_SIZE", 0.1)
    max_training_rows: Optional[int] = (
        _get_env_int("MAX_TRAINING_ROWS", None) if os.getenv("MAX_TRAINING_ROWS") else None
    )
    
    # Fallos de memoria
    dq_sample_rows = _get_env_int("DQ_SAMPLE_ROWS", 10000)

    def dq_sample_size(self) -> Optional[int]:
        if self.dq_sample_rows is None:
            return None
        return self.dq_sample_rows if self.dq_sample_rows > 0 else None


@dataclass(frozen=True)
class PathsConfig:
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    artifacts_dir: Path = ARTIFACTS_DIR
    reports_dir: Path = REPORTS_DIR
    mlruns_dir: Path = MLRUNS_DIR

    def dataset_path(self) -> Path:
        return self.data_dir / "dataset.csv"

    def data_description_path(self) -> Path:
        return self.data_dir / "data_descriptions.csv"

    def baseline_parquet_path(self) -> Path:
        return self.data_dir / "baseline.parquet"

    def new_data_path(self) -> Path:
        return self.data_dir / "new_data.csv"

    def reports_path(self) -> Path:
        return self.reports_dir

    def models_path(self) -> Path:
        return self.artifacts_dir


@dataclass(frozen=True)
class Settings:
    minio: MinIOSettings
    mlflow: MLflowSettings
    postgres: PostgresSettings
    pipeline: PipelineSettings
    paths: PathsConfig


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    paths = PathsConfig()
    return Settings(
        minio=MinIOSettings(),
        mlflow=MLflowSettings(),
        postgres=PostgresSettings(),
        pipeline=PipelineSettings(),
        paths=paths,
    )

# Exports públicos >> from config import *
__all__ = ["get_settings", "Settings"]
