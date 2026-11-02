"""Interpretação: feature importance, permutation importance e SHAP."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def _barh(series, title, fname, color="#264653"):
    fig, ax = plt.subplots(figsize=(7, 6))
    series.sort_values().plot.barh(ax=ax, color=color)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=120)
    plt.close(fig)


def logistic_coefficients(pipe, top=15):
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(pipe.named_steps["model"].coef_[0], index=names)
    top_coef = coef.reindex(coef.abs().sort_values(ascending=False).index)[:top]
    fig, ax = plt.subplots(figsize=(7, 6))
    s = top_coef.sort_values()
    ax.barh(s.index, s.values, color=["#e76f51" if v > 0 else "#2a9d8f" for v in s])
    ax.set_title("Regressão Logística: coeficientes (variáveis padronizadas)\n>0 aumenta chance de maligno")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_coeficientes_logistica.png", dpi=120)
    plt.close(fig)
    return coef


def tree_importance(pipe, top=15):
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    _barh(imp.nlargest(top), "Random Forest: feature importance (impureza de Gini)", "10_importancia_random_forest.png")
    return imp


def permutation(pipe, X, y, top=15, random_state=42):
    r = permutation_importance(pipe, X, y, scoring="recall", n_repeats=30, random_state=random_state, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X.columns)
    _barh(imp.nlargest(top), "Permutation importance (queda no recall, teste)", "11_permutation_importance.png", "#e9c46a")
    return imp


def shap_analysis(pipe, X_background, X_explain, model_kind="linear"):
    """Valores SHAP do modelo escolhido, explicados no conjunto de teste."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = list(prep.get_feature_names_out())
    Xb = pd.DataFrame(prep.transform(X_background), columns=names)
    Xe = pd.DataFrame(prep.transform(X_explain), columns=names, index=X_explain.index)
    if model_kind == "linear":
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(Xb, max_samples=len(Xb)))
        sv = explainer(Xe)
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer(Xe)
        if sv.values.ndim == 3:  # (n, features, classes) -> classe maligna
            sv = sv[:, :, 1]

    shap.summary_plot(sv, Xe, show=False, max_display=15)
    plt.title("SHAP – impacto de cada variável na previsão de MALIGNO")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("SHAP – importância média |SHAP|")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    # explicação individual: o paciente maligno com maior probabilidade prevista
    proba = model.predict_proba(Xe)[:, 1]
    idx = int(np.argmax(proba))
    shap.plots.waterfall(sv[idx], max_display=12, show=False)
    plt.title(f"SHAP – explicação de um paciente (prob. maligno = {proba[idx]:.1%})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_shap_paciente.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=names).sort_values(ascending=False)
    return sv, mean_abs
