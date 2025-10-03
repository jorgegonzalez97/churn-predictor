"""Optimización de parámetros con Optuna"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import roc_auc_score
#from sklearn.model_selection import StratifiedKFold

from src.config import get_settings
from src.features.preprocessing import PreprocessingArtifacts, fit_preprocessing_pipeline
from src.features.schema_inference import DataSchema, infer_schema, load_dataset
from src.io_clients.mlflow_client import MLflowClientWrapper
from src.models.candidate_store import persist_latest_candidate, select_best_candidate
from src.utils.ml_utils import evaluate_predictions, stratified_train_valid_test_split

logger = logging.getLogger(__name__)


@dataclass
class CandidateModel:
    name: str
    params: Dict[str, float | int | str]
    metrics: Dict[str, float]
    estimator_path: Optional[str] = None


def _prepare_datasets(schema: Optional[DataSchema] = None):
    settings = get_settings()
    df = load_dataset()
    schema = schema or infer_schema(df)

    X = df[schema.feature_columns]
    y = df[schema.target].astype(int)

    splits = stratified_train_valid_test_split(X, y, test_size=settings.pipeline.test_size, valid_size=settings.pipeline.validation_size, random_state=settings.pipeline.random_state)

    training_dataset = splits.X_train.copy()
    training_dataset[schema.target] = splits.y_train.values
    pipeline_artifacts = fit_preprocessing_pipeline(training_dataset, schema)
    X_train = pipeline_artifacts.pipeline.transform(splits.X_train)
    X_valid = pipeline_artifacts.pipeline.transform(splits.X_valid)

    return (
        pd.DataFrame(X_train),
        splits.y_train.reset_index(drop=True),
        schema,
        {
            "X_valid": pd.DataFrame(X_valid),
            "y_valid": splits.y_valid.reset_index(drop=True),
        },
        pipeline_artifacts,
    )


def _objective_lightgbm(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, validation: Dict[str, pd.Series]) -> float:
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise RuntimeError("LGBMClassifier no se encuentra instalado")

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
        "random_state": get_settings().pipeline.random_state,
        "objective": "binary",
        "n_jobs": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(validation["X_valid"])[:, 1]
    metrics = evaluate_predictions(validation["y_valid"].values, y_pred)
    trial.set_user_attr("metrics", metrics)
    return 0.5 * metrics["roc_auc"] + 0.5 * metrics["pr_auc"]


def _objective_logistic(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, validation: Dict[str, pd.Series]) -> float:
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(validation["X_valid"])[:, 1]
    metrics = evaluate_predictions(validation["y_valid"].values, y_pred)
    trial.set_user_attr("metrics", metrics)
    return 0.5 * metrics["roc_auc"] + 0.5 * metrics["pr_auc"]

def _objective_xgb(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, validation: Dict[str, pd.Series]) -> float:
    try:
        from xgboost import XGBClassifier
    except Exception: 
        raise RuntimeError("XGBClassifier no se encuentra instalado")
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "logloss",
        "random_state": get_settings().pipeline.random_state,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(validation["X_valid"], validation["y_valid"])], verbose=False)
    y_pred = model.predict_proba(validation["X_valid"])[:, 1]
    metrics = evaluate_predictions(validation["y_valid"].values, y_pred)
    trial.set_user_attr("metrics", metrics)
    return 0.5 * metrics["roc_auc"] + 0.5 * metrics["pr_auc"]

def _objective_catboost(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, validation: Dict[str, pd.Series], schema)-> float:
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception:
        raise RuntimeError("CatBoostClassifier no se encuentra instalado")
    
    # Marca columnas categóricas por dtype para que CatBoost las trate de forma nativa
    cat_features = [X_train.columns.get_loc(col) for col in schema.categorical if col in X_train.columns]

    params = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": get_settings().pipeline.random_state,
        "thread_count": -1,
        "use_best_model": True,
        "allow_writing_files": False,
        "verbose": False,
        "task_type": "CPU",
    }

    # CatBoost solo usa baggin temp con bayesian
    if params["bootstrap_type"] == "Bayesian":
        params.pop("subsample", None)
    else:
        params.pop("bagging_temperature", None)

    train_pool = Pool(X_train, label=y_train, cat_features=cat_features or None)
    valid_pool = Pool(validation["X_valid"], label=validation["y_valid"], cat_features=cat_features or None)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100)

    y_pred = model.predict_proba(validation["X_valid"])[:, 1]
    metrics = evaluate_predictions(validation["y_valid"].values, y_pred)
    trial.set_user_attr("metrics", metrics)
    return 0.5 * metrics["roc_auc"] + 0.5 * metrics["pr_auc"]


def run_hyperparameter_search(schema: Optional[DataSchema] = None) -> Tuple[List[CandidateModel], DataSchema, PreprocessingArtifacts]:
    X_train, y_train, schema, validation, artifacts = _prepare_datasets(schema)
    settings = get_settings()

    candidates: List[CandidateModel] = []
    mlflow_client = MLflowClientWrapper()
    mlflow_callback = MLflowCallback(tracking_uri=get_settings().mlflow.tracking_uri, metric_name="val_score")

    logger.info("Empezando optimización de parámetros con Optuna para el Logistic Regression baseline")
    study = optuna.create_study(direction="maximize", study_name="logreg_baseline")
    study.optimize(
        lambda trial: _objective_logistic(trial, X_train, y_train, validation),
        n_trials=settings.pipeline.optuna_n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
    )
    best_trial = study.best_trial
    candidates.append(CandidateModel(name="logistic_regression", params=best_trial.params, metrics=best_trial.user_attrs["metrics"]))

    logger.info("Empezando optimización de parámetros con Optuna para el LightGBM baseline")
    study = optuna.create_study(direction="maximize", study_name="lightgbm")
    study.optimize(
        lambda trial: _objective_lightgbm(trial, X_train, y_train, validation),
        n_trials=settings.pipeline.optuna_n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
    )
    best_trial = study.best_trial
    candidates.append(CandidateModel(name="lightgbm", params=best_trial.params, metrics=best_trial.user_attrs["metrics"]))

    logger.info("Empezando optimización de parámetros con Optuna para el XGBoost baseline")
    study = optuna.create_study(direction="maximize", study_name="xgboost")
    study.optimize(
        lambda trial: _objective_xgb(trial, X_train, y_train, validation),
        n_trials=settings.pipeline.optuna_n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
    )
    best_trial = study.best_trial
    candidates.append(CandidateModel(name="xgboost", params=best_trial.params, metrics=best_trial.user_attrs["metrics"]))
    
    if False:
        logger.info("Empezando optimización de parámetros con Optuna para el CatBoostClassifie baseliner")
        study = optuna.create_study(direction="maximize", study_name="catboost")
        study.optimize(
            lambda trial: _objective_catboost(trial, X_train, y_train, validation, schema),
            n_trials=settings.pipeline.optuna_n_trials,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
        )
        best_trial = study.best_trial
        candidates.append(CandidateModel(name="catboost", params=best_trial.params, metrics=best_trial.user_attrs["metrics"]))

    with mlflow_client.start_run(run_name="hyperparameter_tuning"):
        summary = {candidate.name: {"params": candidate.params, "metrics": candidate.metrics} for candidate in candidates}
        mlflow_client.log_dict(summary, "tuning/candidates.json")
        aggregate_metrics = {f"best_{candidate.name}_roc_auc": candidate.metrics["roc_auc"] for candidate in candidates}
        mlflow_client.log_metrics(aggregate_metrics)

    try:
        best_candidate = select_best_candidate(candidates)
        persist_latest_candidate(settings, best_candidate)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("No se pudo persistir el mejor candidato: %s", exc)

    return candidates, schema, artifacts


__all__ = ["CandidateModel", "run_hyperparameter_search"]
