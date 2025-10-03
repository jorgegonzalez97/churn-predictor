"""Utilidades para persistir y recuperar el mejor candidato de entrenamiento."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

LATEST_CANDIDATE_FILENAME = "latest_candidate.json"


def _latest_candidate_path(settings) -> Path:
    return settings.paths.models_path() / LATEST_CANDIDATE_FILENAME


def _extract_attr(candidate, attr: str, default=None):
    if hasattr(candidate, attr):
        return getattr(candidate, attr)
    if isinstance(candidate, dict):
        return candidate.get(attr, default)
    return default


def load_latest_candidate(settings, factory = None):
    path = _latest_candidate_path(settings)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc: 
        logger.warning("No se pudo leer el candidato almacenado en %s: %s", path, exc)
        return None

    name = payload.get("name")
    params = payload.get("params")
    metrics = payload.get("metrics", {})
    if not name or params is None:
        logger.warning("El archivo de candidato almacenado está incompleto: %s", path)
        return None

    if not isinstance(metrics, dict):
        metrics = {}

    if factory is None:
        return payload

    try:
        return factory(name=name, params=params, metrics=metrics)
    except Exception as exc: 
        logger.warning("No se pudo reconstruir el candidato almacenado: %s", exc)
        return None


def persist_latest_candidate(settings, candidate) -> None:
    name = _extract_attr(candidate, "name")
    params = _extract_attr(candidate, "params")
    metrics = _extract_attr(candidate, "metrics", {}) or {}

    if not name or params is None:
        logger.warning("No se pudo persistir el candidato: faltan nombre o parámetros")
        return

    payload = {
        "name": name,
        "params": params,
        "metrics": metrics,
        "saved_at": datetime.utcnow().isoformat(),
    }

    path = _latest_candidate_path(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("No se pudo guardar el candidato seleccionado en %s: %s", path, exc)


def select_best_candidate(candidates):
    if not candidates:
        raise ValueError("No hay candidatos para seleccionar")

    def score(candidate) -> float:
        metrics = _extract_attr(candidate, "metrics", {}) or {}
        return 0.5 * float(metrics.get("pr_auc", 0.0)) + 0.5 * float(metrics.get("roc_auc", 0.0))

    best = max(candidates, key=score)
    logger.info("%s seleccionado como mejor modelo (score %.4f)", _extract_attr(best, "name"), score(best))
    return best


__all__ = ["load_latest_candidate", "persist_latest_candidate", "select_best_candidate"]
