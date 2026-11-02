"""Carregamento da base Breast Cancer Wisconsin (Diagnostic).

A base é a mesma publicada no Kaggle (uciml/breast-cancer-wisconsin-data)
e no UCI ML Repository. Se o arquivo ``data/data.csv`` baixado do Kaggle
existir, ele é usado; caso contrário, a cópia que acompanha o scikit-learn
é carregada e convertida para o mesmo formato de colunas do Kaggle.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_CSV = ROOT / "data" / "data.csv"
LOCAL_CSV = ROOT / "data" / "breast_cancer_wisconsin.csv"
TARGET = "diagnosis"


def _sklearn_name_to_kaggle(name: str) -> str:
    """'mean radius' -> 'radius_mean', 'radius error' -> 'radius_se', 'worst radius' -> 'radius_worst'."""
    if name.startswith("mean "):
        return name[5:].replace(" ", "_") + "_mean"
    if name.endswith(" error"):
        return name[:-6].replace(" ", "_") + "_se"
    if name.startswith("worst "):
        return name[6:].replace(" ", "_") + "_worst"
    return name.replace(" ", "_")


def export_sklearn_copy(path: Path = LOCAL_CSV) -> Path:
    """Salva a cópia do scikit-learn em CSV no formato do Kaggle (diagnosis = M/B)."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.data.rename(columns=_sklearn_name_to_kaggle)
    # no scikit-learn: 0 = maligno, 1 = benigno
    df.insert(0, TARGET, raw.target.map({0: "M", 1: "B"}))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_raw() -> pd.DataFrame:
    """Lê a base bruta (Kaggle se disponível, senão a cópia local)."""
    if KAGGLE_CSV.exists():
        return pd.read_csv(KAGGLE_CSV)
    if not LOCAL_CSV.exists():
        export_sklearn_copy()
    return pd.read_csv(LOCAL_CSV)
