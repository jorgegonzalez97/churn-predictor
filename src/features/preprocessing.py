"""Pipeline de preprocesado de variables"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from src.config import get_settings
from src.features.schema_inference import DataSchema, load_dataset, infer_schema
from src.io_clients.minio_client import MinioClient

logger = logging.getLogger(__name__)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Capar outliers para reducir influencias extremas"""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X: np.ndarray, y = None):
        if X.size == 0:
            self.lower_bounds_ = np.array([])
            self.upper_bounds_ = np.array([])
            return self
        self.lower_bounds_ = np.nanquantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Winsorizer debe estar ajustado para poder transformarlo")
        clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return clipped
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            n = 0 if self.lower_bounds_ is None else len(self.lower_bounds_)
            return np.array([f"feature_{i}" for i in range(n)], dtype=object)
        return np.asarray(input_features, dtype=object)


@dataclass
class PreprocessingArtifacts:
    schema: DataSchema
    pipeline: Pipeline
    feature_names: List[str]
    local_path: Path
    minio_uri: Optional[str]


def build_column_transformer(schema: DataSchema) -> ColumnTransformer:
    numeric_pipeline_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
        ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scaler", StandardScaler()),
    ]

    categorical_pipeline_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers = []
    if schema.numerical:
        transformers.append(("numeric", Pipeline(numeric_pipeline_steps), schema.numerical))
    if schema.categorical:
        transformers.append(("categorical", Pipeline(categorical_pipeline_steps), schema.categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_preprocessing_pipeline(schema: DataSchema) -> Pipeline:
    column_transformer = build_column_transformer(schema)
    pipeline = Pipeline(steps=[("preprocess", column_transformer)])
    return pipeline


def fit_preprocessing_pipeline(dataset: pd.DataFrame, schema: Optional[DataSchema] = None, max_rows: Optional[int] = None) -> PreprocessingArtifacts:
    """Ajustar el pipeline de preprocessing y alamcenar artefactos localmente y a MinIO"""

    settings = get_settings()
    if schema is None:
        schema = infer_schema(dataset)

    if max_rows and len(dataset) > max_rows:
        dataset = dataset.sample(n=max_rows, random_state=settings.pipeline.random_state)

    pipeline = build_preprocessing_pipeline(schema)
    X = dataset[schema.feature_columns]
    pipeline.fit(X)

    preprocessed = pipeline.named_steps["preprocess"].transformers_
    feature_names = []
    for name, transformer, cols in preprocessed:
        if hasattr(transformer, "get_feature_names_out"):
            names = list(transformer.get_feature_names_out(cols))
        elif hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
            names = list(transformer.named_steps["encoder"].get_feature_names_out(cols))
        else:
            names = list(cols)
        feature_names.extend(names)

    output_dir = settings.paths.models_path()
    output_dir.mkdir(exist_ok=True, parents=True)
    pipeline_path = output_dir / "preprocessing_pipeline.joblib"
    joblib.dump({"pipeline": pipeline, "schema": schema}, pipeline_path)

    logger.info("Preprocessing pipeline guardada en %s", pipeline_path)

    minio_uri = None
    try:
        client = MinioClient.from_settings()
        object_name = f"pipelines/preprocessing/{pipeline_path.name}"
        minio_uri = client.upload_file(pipeline_path, bucket=settings.minio.bucket_models, object_name=object_name)
        logger.info("Preprocessing pipeline subida a %s", minio_uri)
    except Exception as exc:
        logger.warning("Ha fallado la subida del preprocessing pipeline a MinIO: %s", exc)

    return PreprocessingArtifacts(
        schema=schema,
        pipeline=pipeline,
        feature_names=feature_names,
        local_path=pipeline_path,
        minio_uri=minio_uri,
    )


def load_default_preprocessing(sample_rows: Optional[int] = None) -> PreprocessingArtifacts:
    dataset = load_dataset(sample_rows)
    schema = infer_schema(dataset)
    return fit_preprocessing_pipeline(dataset, schema, max_rows=None)


__all__ = [
    "Winsorizer",
    "PreprocessingArtifacts",
    "build_preprocessing_pipeline",
    "fit_preprocessing_pipeline",
    "load_default_preprocessing",
]
