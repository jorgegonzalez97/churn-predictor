"""Interaccion con tracking y registros de MLflow"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class MLflowRunContext:
    experiment_id: str
    run_id: str
    run_name: Optional[str]


class MLflowClientWrapper:
    def __init__(self) -> None:
        settings = get_settings().mlflow
        mlflow.set_tracking_uri(settings.tracking_uri)
        if settings.registry_uri:
            mlflow.set_registry_uri(settings.registry_uri)
        if settings.experiment_name:
            experiment = mlflow.get_experiment_by_name(settings.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(settings.experiment_name, artifact_location=settings.artifact_location)
                logger.info("Creado experimento de MLflow %s", settings.experiment_name)
            else:
                experiment_id = experiment.experiment_id
        else:
            experiment_id = "0"
        self.experiment_id = experiment_id
        self.client = MlflowClient()

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None ) -> Iterator[MLflowRunContext]:
        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id, tags=tags) as active_run:
            context = MLflowRunContext(experiment_id=active_run.info.experiment_id, run_id=active_run.info.run_id, run_name=run_name)
            logger.debug("Started MLflow run %s", context.run_id)
            yield context
            logger.debug("Finished MLflow run %s", context.run_id)

    def log_params(self, params: Dict[str, str | float | int]) -> None:
        if params:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    # Para JSONs
    def log_dict(self, dictionary: Dict, artifact_file: str) -> None:
        mlflow.log_dict(dictionary, artifact_file)

    # Para otros tags personalizados
    def set_tags(self, tags: Dict[str, str]) -> None:
        if tags:
            mlflow.set_tags(tags)

    def register_model(self, model_uri: str, name: str, run_id: Optional[str] = None) -> str:
        try:
            self.client.create_registered_model(name)
            logger.info("Registro de modelo de MLFlow Creado %s", name)
        except RestException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        active_run = mlflow.active_run()
        resolved_run_id = run_id or (active_run.info.run_id if active_run else None)
        model_version = self.client.create_model_version(name=name, source=model_uri, run_id=resolved_run_id)
        return str(model_version.version)

    # Para ciclo de vida de registry: Prod, Dev, ...
    def transition_stage(self, model_name: str, version: str, stage: str) -> None:
        self.client.transition_model_version_stage( name=model_name, version=version, stage=stage, archive_existing_versions=False )


__all__ = ["MLflowClientWrapper", "MLflowRunContext"]
