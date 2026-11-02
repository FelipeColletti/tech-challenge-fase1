# Dados

**Breast Cancer Wisconsin (Diagnostic)** – 569 pacientes, 30 variáveis numéricas, alvo `diagnosis` (M = maligno, B = benigno).

- Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- UCI: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

`breast_cancer_wisconsin.csv` é a mesma base, exportada da cópia que acompanha o scikit-learn
(`sklearn.datasets.load_breast_cancer`) com os nomes de colunas do Kaggle. Se você baixar o `data.csv`
do Kaggle e colocá-lo nesta pasta, o projeto passa a usá-lo automaticamente (as colunas `id` e
`Unnamed: 32` são removidas na limpeza).
