"""Limpeza e pipeline de pré-processamento."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET

RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas sem informação, duplicatas e valores inconsistentes.

    - ``id`` é apenas um identificador (não ajuda no diagnóstico);
    - ``Unnamed: 32`` é uma coluna vazia que vem no CSV do Kaggle;
    - medidas morfológicas não podem ser negativas: viram NaN e são imputadas no pipeline;
    - o alvo M/B é convertido para 1 (maligno) / 0 (benigno).
    """
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c == "id" or c.startswith("Unnamed")])
    df = df.drop_duplicates()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper().map({"M": 1, "B": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    features = [c for c in df.columns if c != TARGET]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df[features] = df[features].mask(df[features] < 0, np.nan)
    return df


def split(df: pd.DataFrame, val_size=0.2, test_size=0.2, random_state=RANDOM_STATE):
    """Separa treino / validação / teste (60/20/20), estratificado pelo diagnóstico."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(numeric_cols, categorical_cols=()):
    """Pipeline: imputação pela mediana + padronização (numéricas) e one-hot (categóricas).

    A base escolhida só tem variáveis numéricas, mas o pipeline já trata
    categóricas para poder ser reaproveitado com outras bases.
    """
    from sklearn.preprocessing import OneHotEncoder

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers = [("num", numeric, list(numeric_cols))]
    if len(categorical_cols):
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical, list(categorical_cols)))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
