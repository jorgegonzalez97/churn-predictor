"""Ingesta de ddatos y ajuste de schema"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.config import get_settings


@dataclass(frozen=True)
class DataSchema:
    """Describir la estructura de los datos

    Atributos:
        numerical: Variable tratada como numérica
        categorical: Variable tratada como categórica one-hot.
        high_cardinality:  Variable Categórica excluída de one-hot encoding.
        target: Columna Target
        id_column: Columna identificadora.
    """

    numerical: List[str]
    categorical: List[str]
    high_cardinality: List[str]
    target: str
    id_column: Optional[str] = None

    @property
    def feature_columns(self) -> List[str]:
        excluded = {self.target}
        if self.id_column:
            excluded.add(self.id_column)
        excluded.update(self.high_cardinality)
        return [col for col in self.numerical + self.categorical if col not in excluded]

    @property
    def all_columns(self) -> List[str]:
        cols: List[str] = []
        if self.id_column:
            cols.append(self.id_column)
        cols.extend(self.feature_columns)
        cols.append(self.target)
        return cols


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sin dataset de entrada en {path}")
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")

# Cargar dataset original respecto a la configdel proyecto
def load_dataset(sample_rows: Optional[int] = None) -> pd.DataFrame:
    settings = get_settings()
    df = _load_csv(settings.paths.dataset_path())
    if sample_rows:
        return df.sample(n=sample_rows, random_state=settings.pipeline.random_state)
    return df

# Cargar dicccionario con definiciones de los datos
def load_data_description() -> Dict[str, str]:
    settings = get_settings()
    description_path = settings.paths.data_description_path()
    if not description_path.exists():
        return {}
    df = pd.read_csv(description_path, sep=";", encoding="utf-8")
    return {
        str(row["Variable"]).strip(): str(row["Significado"]).strip()
        for _, row in df.iterrows()
        if "Variable" in row and "Significado" in row
    }


def infer_schema(dataset: pd.DataFrame) -> DataSchema:
    """Análisis de estructura de los datos automática basada en reglas:
            Columnas numéricas con baja cardinalidad codificadas como categóricas
            Varible ID no se utilizan en el entreno"""

    settings = get_settings()
    target_col = settings.pipeline.target_column
    id_col = settings.pipeline.id_column if settings.pipeline.id_column in dataset.columns else None

    id_candidates = dataset.columns[dataset.columns.str.lower().str.contains("id")]
    if id_col and id_col not in dataset.columns:
        id_col = id_candidates[0] if len(id_candidates) > 0 else None

    if target_col not in dataset.columns:
        raise KeyError(f"La columna target '{target_col}' no está presente en el dataset")

    numeric_cols = []
    categorical_cols = []
    high_cardinality = []
    
    #print(dataset.columns)
    all_feat = ['rev_Mean', 'mou_Mean', 'totmrc_Mean','da_Mean','ovrmou_Mean','datovr_Mean','roam_Mean','change_mou', 'change_rev',
        'unan_vce_Mean', 'plcd_dat_Mean','custcare_Mean','ccrndmou_Mean','inonemin_Mean','threeway_Mean','mou_rvce_Mean',
        'owylis_vce_Mean', 'mouowylisv_Mean','iwylis_vce_Mean','mouiwylisv_Mean','mou_peav_Mean','mou_pead_Mean','mou_opkv_Mean','mou_opkd_Mean',
        'drop_blk_Mean', 'callwait_Mean','months','uniqsubs','actvsubs','new_cell','crclscod','asl_flag','totrev','adjqty','avg6rev',
        'prizm_social_one','area','dualband','refurb_new','hnd_price','models','hnd_webcap','lor','marital','adults','income','ethnic','creditcd','eqpdays','Customer_ID']

    for column in all_feat:
        if column == target_col:
            continue
        if id_col and column == id_col:
            high_cardinality.append(column)
            continue

        series = dataset[column]
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
            if unique_ratio < 0.05 and series.nunique(dropna=True) < 20:
                categorical_cols.append(column)
            else:
                numeric_cols.append(column)
        else:
            nunique = series.nunique(dropna=True)
            if nunique > 50:
                high_cardinality.append(column)
            else:
                categorical_cols.append(column)

    numeric_cols = sorted(set(numeric_cols))
    categorical_cols = sorted(set(categorical_cols))
    high_cardinality = sorted(set(high_cardinality))

    return DataSchema(
        numerical=numeric_cols,
        categorical=categorical_cols,
        high_cardinality=high_cardinality,
        target=target_col,
        id_column=id_col,
    )


def infer_schema_from_path(path: Optional[Path] = None) -> DataSchema:
    dataset = _load_csv(path or get_settings().paths.dataset_path())
    return infer_schema(dataset)


__all__ = ["DataSchema", "load_dataset", "load_data_description", "infer_schema", "infer_schema_from_path"]
