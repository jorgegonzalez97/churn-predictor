"""DAG de Airflow para preparar datos y ejecutar la busqueda de hiperparametros"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.models.tuning import CandidateModel

logger = logging.getLogger(__name__)


def _to_serializable(value: Any) -> Any:
    """Devuelve un objeto apto para JSON (maneja numpy/pandas y Paths)."""

    # Números nativos, cadenas y None ya son serializables
    if value is None or isinstance(value, (int, float, bool, str)):
        return value

    # Objetos numpy/pandas suelen tener método item() para obtener el escalar
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    # Secuencias: convertimos elemento a elemento
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(elem) for elem in value]

    # Diccionarios: serializamos cada valor
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}

    # Rutas a string (evita errores con Pathlib)
    if isinstance(value, Path):
        return str(value)

    # Cualquier otro objeto lo convertimos a cadena como último recurso
    return str(value)


def _candidate_to_dict(candidate: "CandidateModel") -> Dict[str, object]:
    return {
        "name": candidate.name,
        "params": _to_serializable(candidate.params),
        "metrics": _to_serializable(candidate.metrics),
    }


@dag(
    dag_id="dag_model_selection_tuning",
    schedule="@monthly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"owner": "mlops"},
    tags=["model-selection", "tuning"],
)
def model_selection_tuning_pipeline():
    """Pipeline mensual para preparar datos y evaluar candidatos."""

    @task
    def prepare_data() -> Dict[str, object]:
        from src.features.preprocessing import load_default_preprocessing

        artifacts = load_default_preprocessing(sample_rows=2000)
        logger.info("Artefactos preprocesados generados en %s", artifacts.local_path)
        return {
            "pipeline_path": str(artifacts.local_path),
            "schema_features": artifacts.schema.feature_columns,
        }

    @task
    def hyperparameter_tuning(_: Dict[str, object]) -> Dict[str, object]:
        from src.models.tuning import run_hyperparameter_search

        candidates, schema, _ = run_hyperparameter_search()
        logger.info("%d modelos candidatos evaluados", len(candidates))
        return {
            "candidates": [_candidate_to_dict(candidate) for candidate in candidates],
            "feature_count": len(schema.feature_columns),
        }


    tuning_info = hyperparameter_tuning(prepare_data())


model_selection_tuning_dag = model_selection_tuning_pipeline()
