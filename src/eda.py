"""Análise exploratória: estatísticas descritivas e gráficos."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import TARGET

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
LABELS = {0: "Benigno", 1: "Maligno"}
PALETTE = {"Benigno": "#2a9d8f", "Maligno": "#e76f51"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)


def summary(df: pd.DataFrame) -> dict:
    return {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "valores_ausentes": int(df.isna().sum().sum()),
        "duplicatas": int(df.duplicated().sum()),
        "distribuicao_alvo": df[TARGET].value_counts().rename(LABELS).to_dict(),
    }


def plot_target(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].map(LABELS).value_counts()
    ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f"{v} ({v / counts.sum():.0%})", ha="center")
    ax.set_title("Distribuição do diagnóstico")
    ax.set_ylabel("Pacientes")
    _save(fig, "01_distribuicao_alvo.png")


def plot_distributions(df, features):
    data = df.assign(Diagnóstico=df[TARGET].map(LABELS))
    n = len(features)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2))
    for ax, feat in zip(axes.flat, features):
        sns.histplot(data, x=feat, hue="Diagnóstico", kde=True, palette=PALETTE,
                     stat="density", common_norm=False, ax=ax)
        ax.set_title(feat)
        ax.set_xlabel("")
    for ax in list(axes.flat)[n:]:
        ax.remove()
    _save(fig, "02_distribuicoes.png")


def plot_boxplots(df, features):
    data = df.assign(Diagnóstico=df[TARGET].map(LABELS))
    long = data.melt(id_vars="Diagnóstico", value_vars=features)
    # padroniza para caber no mesmo eixo
    long["value"] = long.groupby("variable")["value"].transform(lambda s: (s - s.mean()) / s.std())
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(long, x="variable", y="value", hue="Diagnóstico", palette=PALETTE, ax=ax, fliersize=2)
    ax.set_xlabel("")
    ax.set_ylabel("Valor padronizado (z-score)")
    ax.set_title("Variáveis *_mean por diagnóstico (outliers aparecem como pontos)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "03_boxplots.png")


def plot_correlation(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Matriz de correlação (Pearson)")
    _save(fig, "04_correlacao.png")

    target_corr = corr[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = ["#e76f51" if v > 0 else "#2a9d8f" for v in target_corr]
    ax.barh(target_corr.index, target_corr.values, color=colors)
    ax.set_title("Correlação de cada variável com o diagnóstico (1 = maligno)")
    _save(fig, "05_correlacao_alvo.png")
    return corr, target_corr


def high_correlation_pairs(corr, threshold=0.9):
    cols = [c for c in corr.columns if c != TARGET]
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(corr.loc[a, b]) >= threshold:
                pairs.append((a, b, round(float(corr.loc[a, b]), 3)))
    return sorted(pairs, key=lambda p: -abs(p[2]))
