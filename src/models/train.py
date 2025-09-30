"""Entreno evaluación y almacenado de modelo final"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline

from src.config import get_settings
from src.features.preprocessing import PreprocessingArtifacts, fit_preprocessing_pipeline
from src.features.schema_inference import DataSchema, infer_schema, load_dataset
from src.io_clients.minio_client import MinioClient
from src.io_clients.mlflow_client import MLflowClientWrapper
from src.models.tuning import CandidateModel, run_hyperparameter_search
from src.utils.ml_utils import DatasetSplits, evaluate_predictions, stratified_train_valid_test_split
from src.utils.plotting import plot_calibration, plot_lift, plot_precision_recall, plot_roc, save_figure

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    run_id: str
    model_name: str
    metrics: Dict[str, float]
    local_model_path: Path
    minio_uri: Optional[str]
    mlflow_run_id: str
    mlflow_model_version: Optional[str]


MODEL_REGISTRY_NAME = "churn-risk-model"
PROMOTE_MIN_ROC = float(os.getenv("PROMOTE_MIN_ROC", "0.7"))
PROMOTE_MIN_PR = float(os.getenv("PROMOTE_MIN_PR", "0.35"))


def _instantiate_estimator(candidate: CandidateModel):
    if candidate.name == "logistic_regression":
        penalty = candidate.params.get("penalty", "l2")
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        return LogisticRegression(penalty=penalty, C=float(candidate.params.get("C", 1.0)), solver=solver, max_iter=2000)
    if candidate.name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception:
            raise RuntimeError("LGBMClassifier no está disponible")
        params = {key: value for key, value in candidate.params.items() if key not in {"random_state", "objective"}}
        params.update({"objective": "binary", "random_state": get_settings().pipeline.random_state, "n_jobs": -1})
        return LGBMClassifier(**params)
    if candidate.name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception:
            raise RuntimeError("XGBClassifier no está disponible")
        params = dict(candidate.params)
        params.update({"random_state": get_settings().pipeline.random_state, "use_label_encoder": False, "eval_metric": "logloss"})
        return XGBClassifier(**params)
    if candidate.name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception:
            raise RuntimeError("CatBoost no está disponible")

        drop_keys = {"random_state", "eval_metric", "loss_function", "n_jobs"}
        params = {k: v for k, v in candidate.params.items() if k not in drop_keys}

        params.update({
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": get_settings().pipeline.random_state,
            "thread_count": -1,
            "use_best_model": True,
            "allow_writing_files": False,
            "verbose": False,
            "task_type": "CPU"
        })

        # Si en el tuning se fijó bootstrap_type=Bayesian, CatBoost ignora subsample
        if params.get("bootstrap_type", None) == "Bayesian":
            params.pop("subsample", None)

        return CatBoostClassifier(**params)
    raise ValueError(f"Modelo candidato desconocido: {candidate.name}")


def _select_best_candidate(candidates: list[CandidateModel]) -> CandidateModel:
    def score(candidate: CandidateModel) -> float:
        metrics = candidate.metrics
        return 0.5 * metrics.get("pr_auc", 0.0) + 0.5 * metrics.get("roc_auc", 0.0)

    best = max(candidates, key=score)
    logger.info(" %s seleccionado como mejor modelo (score %.4f)", best.name, score(best))
    return best


def _prepare_training_entities(schema: Optional[DataSchema] = None) -> tuple[DatasetSplits, DataSchema, PreprocessingArtifacts]:
    settings = get_settings()
    df = load_dataset()
    schema = schema or infer_schema(df)

    X = df[schema.feature_columns]
    y = df[schema.target].astype(int)

    splits = stratified_train_valid_test_split(X, y, test_size=settings.pipeline.test_size, valid_size=settings.pipeline.validation_size, random_state=settings.pipeline.random_state)

    X_train_full = pd.concat([splits.X_train, splits.X_valid])
    y_train_full = pd.concat([splits.y_train, splits.y_valid])
    train_dataset = X_train_full.copy()
    train_dataset[schema.target] = y_train_full.values
    preproc_artifacts = fit_preprocessing_pipeline(train_dataset, schema)

    metadata = {
        "feature_names": preproc_artifacts.feature_names,
        "schema": schema,
    }
    return splits, metadata, preproc_artifacts


def _generate_and_log_figures(y_test: pd.Series, y_pred_proba: np.ndarray, feature_names: list[str], estimator, X_test_preprocessed: np.ndarray, output_dir: Path, mlflow_client: MLflowClientWrapper):
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_fig = plot_roc(y_test.values, y_pred_proba)
    pr_fig = plot_precision_recall(y_test.values, y_pred_proba)
    lift_fig = plot_lift(y_test.values, y_pred_proba)
    calibration_fig = plot_calibration(y_test.values, y_pred_proba)

    roc_path = save_figure(roc_fig, output_dir / "roc_curve.png")
    pr_path = save_figure(pr_fig, output_dir / "pr_curve.png")
    lift_path = save_figure(lift_fig, output_dir / "lift_curve.png")
    calibration_path = save_figure(calibration_fig, output_dir / "calibration_curve.png")

    for path in [roc_path, pr_path, lift_path, calibration_path]:
        mlflow_client.log_artifact(path, artifact_path="figures")

    # Feature importance chart
    importance_path = output_dir / "feature_importance.json"
    importances= {}
    if hasattr(estimator, "feature_importances_"):
        importances = {feature: float(importance) for feature, importance in zip(feature_names, estimator.feature_importances_)}
    elif hasattr(estimator, "coef_"):
        coefs = estimator.coef_[0]
        importances = {feature: float(abs(coef)) for feature, coef in zip(feature_names, coefs)}
    importance_path.write_text(json.dumps(importances, indent=2), encoding="utf-8")
    mlflow_client.log_artifact(importance_path, artifact_path="explainability")

    # SHAP summary
    try:
        sample_size = min(500, X_test_preprocessed.shape[0])
        background_idx = np.random.choice(X_test_preprocessed.shape[0], size=sample_size, replace=False)
        background = X_test_preprocessed[background_idx]
        shap_values_path = output_dir / "shap_summary.png"
        if hasattr(estimator, "predict_proba"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(background)
            shap.summary_plot(shap_values, features=background, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(shap_values_path)
            plt.close()
            mlflow_client.log_artifact(shap_values_path, artifact_path="explainability")
    except Exception as exc: 
        logger.warning("No se pudo calcular el resumen SHAP: %s", exc)


def train_and_register_model() -> TrainingResult:
    settings = get_settings()
    candidates, schema, _preproc_artifacts = run_hyperparameter_search()
    best_candidate = _select_best_candidate(candidates)

    splits, metadata, preproc_artifacts = _prepare_training_entities(schema)
    preprocessor = preproc_artifacts.pipeline.named_steps["preprocess"]
    model_pipeline = SkPipeline(steps=[("preprocess", preprocessor), ("estimator", _instantiate_estimator(best_candidate))])
    X_train_full = pd.concat([splits.X_train, splits.X_valid])
    y_train_full = pd.concat([splits.y_train, splits.y_valid])
    model_pipeline.fit(X_train_full, y_train_full)
    y_test = splits.y_test.reset_index(drop=True)
    y_pred_proba = model_pipeline.predict_proba(splits.X_test)[:, 1]
    feature_names = metadata["feature_names"]
    X_test_preprocessed = preprocessor.transform(splits.X_test)

    metrics = evaluate_predictions(y_test.values, y_pred_proba)

    run_identifier = uuid.uuid4().hex[:12]
    model_dir = settings.paths.models_path()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"churn_model_{run_identifier}.joblib"
    packed_model = {
        "pipeline": model_pipeline,
        "schema": metadata["schema"],
        "feature_names": feature_names,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
    }
    joblib.dump(packed_model, model_path)

    mlflow_client = MLflowClientWrapper()
    with mlflow_client.start_run(run_name="train_final_model", tags={"model": best_candidate.name}) as mlflow_run:
        mlflow_client.log_params({f"best_{best_candidate.name}_{key}": value for key, value in best_candidate.params.items()})
        mlflow_client.log_metrics(metrics)
        mlflow_client.log_artifact(model_path, artifact_path="model")
        _generate_and_log_figures(y_test, y_pred_proba, feature_names, model_pipeline.named_steps["estimator"], X_test_preprocessed, model_dir , "figures" / run_identifier, mlflow_client)
        mlflow_client.set_tags({"stage": "training", "model_name": best_candidate.name, "run_identifier": run_identifier})

        try:
            import mlflow.sklearn

            mlflow.sklearn.log_model(model_pipeline, artifact_path="model_pipeline")
            model_uri = f"runs:/{mlflow_run.run_id}/model_pipeline"
        except Exception as exc:
            logger.warning("No se consiguió loggear el sklearn pipeline en MLflow: %s", exc)
            model_uri = f"runs:/{mlflow_run.run_id}/model"
        mlflow_model_version = None
        try:
            version = mlflow_client.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME, run_id=mlflow_run.run_id)
            if metrics["roc_auc"] >= PROMOTE_MIN_ROC and metrics["pr_auc"] >= PROMOTE_MIN_PR:
                mlflow_client.transition_stage(MODEL_REGISTRY_NAME, version, stage="Production")
            else:
                mlflow_client.transition_stage(MODEL_REGISTRY_NAME, version, stage="Staging")
            mlflow_model_version = version
        except Exception as exc: 
            logger.warning("Algo falló con el registro del modelo: %s", exc)

    minio_client = MinioClient.from_settings()
    object_name = f"models/churn/{run_identifier}/model.joblib"
    minio_uri = None
    try:
        minio_uri = minio_client.upload_file(model_path, bucket=settings.minio.bucket_models, object_name=object_name)
    except Exception as exc: 
        logger.warning("Falló la subida del modelo a MinIO: %s", exc)

    return TrainingResult(
        run_id=run_identifier,
        model_name=best_candidate.name,
        metrics=metrics,
        local_model_path=model_path,
        minio_uri=minio_uri,
        mlflow_run_id=mlflow_run.run_id,
        mlflow_model_version=mlflow_model_version,
    )


__all__ = ["train_and_register_model", "TrainingResult"]
