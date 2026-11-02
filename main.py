"""Executa todo o fluxo do Tech Challenge Fase 1.

    python main.py

Gera gráficos em results/figures, tabelas em results/ e um resumo em results/metrics.json.
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd

from src import eda, interpretation, modeling
from src.data import TARGET, load_raw
from src.preprocessing import clean, split

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 90)
RESULTS = Path(__file__).resolve().parent / "results"


def main():
    print("=" * 70, "\n1) CARREGAMENTO E EXPLORAÇÃO\n" + "=" * 70)
    raw = load_raw()
    print(f"Base bruta: {raw.shape[0]} linhas x {raw.shape[1]} colunas")
    df = clean(raw)
    info = eda.summary(df)
    print(info)
    desc = df.describe().T
    desc.to_csv(RESULTS / "estatisticas_descritivas.csv")
    print(desc[["mean", "std", "min", "50%", "max"]].head(10))
    print("\nMédia por diagnóstico (variáveis *_mean):")
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    print(df.groupby(TARGET)[mean_cols].mean().T.rename(columns={0: "Benigno", 1: "Maligno"}))

    eda.plot_target(df)
    eda.plot_distributions(df, mean_cols)
    eda.plot_boxplots(df, mean_cols)
    corr, target_corr = eda.plot_correlation(df)
    pairs = eda.high_correlation_pairs(corr)
    print("\nTop correlações com o diagnóstico:\n", target_corr.sort_values(ascending=False).head(8))
    print(f"\nPares de variáveis com |r| >= 0.9: {len(pairs)} (ex.: {pairs[:3]})")

    print("\n" + "=" * 70, "\n2) SEPARAÇÃO TREINO / VALIDAÇÃO / TESTE\n" + "=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    for nome, y in [("treino", y_train), ("validação", y_val), ("teste", y_test)]:
        print(f"{nome:10s}: {len(y):3d} amostras | {y.mean():.1%} malignos")

    print("\n" + "=" * 70, "\n3) MODELAGEM (ajuste de hiperparâmetros na VALIDAÇÃO)\n" + "=" * 70)
    models, val_df = modeling.tune_on_validation(X_train, y_train, X_val, y_val)
    print(val_df)
    val_df.to_csv(RESULTS / "resultados_validacao.csv")

    best_name = (val_df.sort_values(["val_recall", "val_f1", "val_roc_auc"], ascending=False).index[0])
    print(f"\nModelo escolhido pela validação: {best_name}")

    print("\n" + "=" * 70, "\n4) AVALIAÇÃO NO TESTE (dados nunca vistos)\n" + "=" * 70)
    test_df = modeling.evaluate_on_test(models, X_test, y_test)
    print(test_df)
    test_df.to_csv(RESULTS / "resultados_teste.csv")
    modeling.plot_confusion_matrices(models, X_test, y_test)
    modeling.plot_roc(models, X_test, y_test)
    modeling.plot_metric_comparison(test_df)

    best = models[best_name]
    thr, thr_df = modeling.threshold_analysis(best, X_val, y_val, X_test, y_test)
    print(f"\nLimiar para recall >= 98% na validação: {thr}")
    print(thr_df)
    thr_df.to_csv(RESULTS / "analise_limiar.csv")

    print("\n" + "=" * 70, "\n5) INTERPRETAÇÃO\n" + "=" * 70)
    coef = interpretation.logistic_coefficients(models["Regressão Logística"])
    rf_imp = interpretation.tree_importance(models["Random Forest"])
    perm = interpretation.permutation(best, X_test, y_test)
    kind = "linear" if best_name == "Regressão Logística" else "tree"
    if best_name == "KNN":  # SHAP exato não se aplica a KNN; explica-se a Random Forest
        best_for_shap, kind = models["Random Forest"], "tree"
    elif best_name in ("Árvore de Decisão", "Random Forest", "Regressão Logística"):
        best_for_shap = best
    _, shap_imp = interpretation.shap_analysis(best_for_shap, X_train, X_test, kind)
    print("Top 10 SHAP:\n", shap_imp.head(10))
    print("\nTop 10 Random Forest:\n", rf_imp.sort_values(ascending=False).head(10))

    RESULTS.joinpath("models").mkdir(exist_ok=True)
    joblib.dump(best, RESULTS / "models" / "melhor_modelo.joblib")

    modeling.save_json({
        "resumo_dados": info,
        "pares_alta_correlacao": pairs,
        "modelo_escolhido": best_name,
        "validacao": val_df.reset_index().to_dict(orient="records"),
        "teste": test_df.reset_index().to_dict(orient="records"),
        "limiar_recall_98": thr,
        "analise_limiar": thr_df.reset_index().to_dict(orient="records"),
        "top_shap": shap_imp.head(10).round(4).to_dict(),
        "top_random_forest": rf_imp.sort_values(ascending=False).head(10).round(4).to_dict(),
        "top_permutation": perm.sort_values(ascending=False).head(10).round(4).to_dict(),
    }, RESULTS / "metrics.json")
    print(f"\nResultados salvos em {RESULTS}")


if __name__ == "__main__":
    main()
