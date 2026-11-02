"""Treinamento, seleção no conjunto de validação e avaliação no teste."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import RANDOM_STATE, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"

# Cada modelo com uma pequena grade de hiperparâmetros, escolhidos pela validação.
MODELS = {
    "Regressão Logística": (
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]},
    ),
    "Árvore de Decisão": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [3, 4, 5, 8, None], "min_samples_leaf": [1, 5, 10], "class_weight": [None, "balanced"]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"n_estimators": [300], "max_depth": [None, 6], "min_samples_leaf": [1, 3], "class_weight": [None, "balanced"]},
    ),
}


def metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def make_pipeline(estimator, feature_cols):
    return Pipeline([("prep", build_preprocessor(feature_cols)), ("model", estimator)])


def tune_on_validation(X_train, y_train, X_val, y_val):
    """Para cada técnica, testa a grade no conjunto de VALIDAÇÃO.

    Critério: maior recall (não deixar câncer passar) e, no empate, maior F1.
    """
    results, best = [], {}
    cols = list(X_train.columns)
    for name, (est, grid) in MODELS.items():
        best_score, best_pipe, best_params = None, None, None
        for params in ParameterGrid(grid):
            pipe = make_pipeline(clone(est).set_params(**params), cols).fit(X_train, y_train)
            proba = pipe.predict_proba(X_val)[:, 1]
            m = metrics(y_val, (proba >= 0.5).astype(int), proba)
            score = (round(m["recall"], 4), round(m["f1"], 4), round(m["roc_auc"], 4))
            if best_score is None or score > best_score:
                best_score, best_pipe, best_params, best_m = score, pipe, params, m
        best[name] = best_pipe
        results.append({"modelo": name, "melhores_parametros": best_params,
                        **{f"val_{k}": v for k, v in best_m.items()}})
    return best, pd.DataFrame(results).set_index("modelo")


def evaluate_on_test(models: dict, X_test, y_test):
    rows = []
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({"modelo": name, **metrics(y_test, (proba >= 0.5).astype(int), proba)})
    return pd.DataFrame(rows).set_index("modelo")


def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4))
    for ax, (name, pipe) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, display_labels=["Benigno", "Maligno"],
                                              cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(name)
    fig.suptitle("Matrizes de confusão no conjunto de TESTE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_matrizes_confusao.png", dpi=120)
    plt.close(fig)


def plot_roc(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in models.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("Curva ROC no conjunto de TESTE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_curva_roc.png", dpi=120)
    plt.close(fig)


def plot_metric_comparison(test_df):
    ax = test_df[["accuracy", "recall", "precision", "f1", "roc_auc"]].plot.bar(figsize=(10, 5), rot=0)
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Comparação de métricas no TESTE")
    ax.legend(loc="lower right", ncol=5)
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / "08_comparacao_metricas.png", dpi=120)
    plt.close(ax.figure)


def threshold_analysis(pipe, X_val, y_val, X_test, y_test, min_recall=0.98):
    """Escolhe, na VALIDAÇÃO, o maior limiar que mantém recall >= min_recall e aplica no teste."""
    proba_val = pipe.predict_proba(X_val)[:, 1]
    chosen = 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        if recall_score(y_val, (proba_val >= t).astype(int)) >= min_recall:
            chosen = round(float(t), 2)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    rows = []
    for t in sorted({0.5, chosen}):
        rows.append({"limiar": t, **metrics(y_test, (proba_test >= t).astype(int), proba_test)})
    return chosen, pd.DataFrame(rows).set_index("limiar")


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
