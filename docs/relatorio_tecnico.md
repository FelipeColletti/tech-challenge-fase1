# Relatório Técnico – Tech Challenge Fase 1
**Apoio ao diagnóstico de câncer de mama com Machine Learning**

## 1. Problema e dados

O hospital precisa acelerar a triagem de exames. Escolhemos o problema de **classificar tumores de mama como malignos
ou benignos** a partir de exames de punção aspirativa por agulha fina (PAAF), usando a base pública
**Breast Cancer Wisconsin (Diagnostic)** (UCI / Kaggle).

- 569 pacientes: 357 benignos (62,7%) e 212 malignos (37,3%).
- 30 variáveis numéricas: 10 características dos núcleos celulares (raio, textura, perímetro, área, suavidade,
  compacidade, concavidade, pontos côncavos, simetria, dimensão fractal), cada uma como média (`_mean`),
  erro padrão (`_se`) e pior valor (`_worst`).
- Sem valores ausentes nem duplicatas.

## 2. Exploração dos dados

- As escalas variam muito (ex.: `area_mean` de ~143 a ~2501; `smoothness_mean` ~0,05 a 0,16) → necessidade de padronização.
- Distribuições com assimetria à direita e outliers (tumores grandes), clinicamente plausíveis → mantidos.
- Malignos têm valores bem maiores de tamanho (raio/perímetro/área) e de irregularidade do contorno
  (concavidade, pontos côncavos). `fractal_dimension_mean`, `smoothness_mean` e `symmetry_mean` separam pouco as classes.
- Correlação com o diagnóstico: `concave_points_worst` (0,79), `perimeter_worst` (0,78), `concave_points_mean` (0,78), `radius_worst` (0,78).
- Multicolinearidade forte: 21 pares com |r| ≥ 0,9 (ex.: `radius_mean` × `perimeter_mean` r = 0,998).

Gráficos: `results/figures/01` a `05`.

## 3. Estratégias de pré-processamento

1. **Limpeza**: remoção de `id` e da coluna vazia `Unnamed: 32` (CSV do Kaggle), remoção de duplicatas,
   conversão do alvo M/B → 1/0, conversão das variáveis para numérico e tratamento de valores inconsistentes
   (medidas negativas viram ausentes).
2. **Valores ausentes**: imputação pela **mediana** (robusta a outliers). A base não tem ausentes, mas o pipeline está pronto para dados reais.
3. **Escalonamento**: `StandardScaler` (média 0, desvio 1), essencial para Regressão Logística e KNN.
4. **Categóricas**: o pipeline aplica one-hot encoding se houver variáveis categóricas (não há nesta base).
5. **Pipeline do scikit-learn** (`ColumnTransformer` + modelo), ajustado **somente no treino**, evitando vazamento de dados.
6. **Multicolinearidade**: mantivemos todas as variáveis e usamos regularização L2 na Regressão Logística; árvores não são afetadas.

## 4. Modelos usados e por quê

| Modelo | Motivo | Hiperparâmetros testados |
|---|---|---|
| Regressão Logística | Linear, interpretável, probabilidades bem calibradas | `C` ∈ {0,01; 0,1; 1; 10}, `class_weight` |
| Árvore de Decisão | Regras explicáveis ao médico | `max_depth`, `min_samples_leaf`, `class_weight` |
| KNN | Não paramétrico, baseado em similaridade entre pacientes | `n_neighbors` ∈ {3…15}, `weights` |
| Random Forest | Robusto a outliers e colinearidade, feature importance | `max_depth`, `min_samples_leaf`, `class_weight` |

**Separação**: 60% treino (341), 20% validação (114), 20% teste (114), estratificada.
Os hiperparâmetros e o modelo final foram escolhidos **na validação**, pelo critério recall → F1 → AUC.
O teste foi usado uma única vez, no final.

## 5. Resultados

**Validação (melhor configuração de cada modelo)**

| Modelo | Parâmetros | Recall | F1 | AUC |
|---|---|---|---|---|
| Regressão Logística | C=1, balanced | 0,977 | 0,966 | 0,996 |
| Árvore de Decisão | max_depth=8, balanced | 0,977 | 0,944 | 0,960 |
| KNN | k=7, distance | 0,930 | 0,964 | 0,997 |
| Random Forest | 300 árvores | 0,953 | 0,965 | 0,992 |

**Teste (114 pacientes, 42 malignos)**

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Cânceres não detectados |
|---|---|---|---|---|---|---|
| **Regressão Logística** | **0,974** | 0,953 | **0,976** | **0,965** | 0,995 | **1** |
| Árvore de Decisão | 0,956 | 0,951 | 0,929 | 0,940 | 0,950 | 3 |
| KNN | 0,947 | 0,974 | 0,881 | 0,925 | 0,984 | 5 |
| Random Forest | 0,956 | 0,974 | 0,905 | 0,938 | 0,997 | 4 |

**Escolha da métrica.** Em diagnóstico de câncer, um **falso negativo** atrasa o tratamento e é muito mais grave que um
falso positivo (que gera exames extras, mas é desfeito pela biópsia). Por isso o **recall da classe maligna** é a métrica
principal. A acurácia sozinha engana com classes desbalanceadas; o F1 equilibra recall e precisão; a AUC mede a separação
das classes em qualquer limiar.

**Modelo escolhido: Regressão Logística**, que encontrou 41 dos 42 cânceres com apenas 2 falsos positivos.

**Limiar de decisão.** Na validação, o maior limiar que garantiu recall ≥ 98% foi 0,15. No teste, ele não recuperou o
único câncer perdido (probabilidade prevista ~10%) e aumentou os falsos positivos (precisão 0,95 → 0,84). O limiar deve
ser definido com a equipe clínica conforme o custo aceitável de exames adicionais.

## 6. Interpretação

- **Coeficientes da Regressão Logística** (`09`), **feature importance da Random Forest** (`10`),
  **permutation importance** (`11`) e **SHAP** (`12`–`14`).
- Todos apontam **tamanho do tumor** (raio, perímetro, área, em especial os valores *worst*) e **irregularidade do
  contorno** (pontos côncavos, concavidade) como os principais sinais de malignidade, o que coincide com critérios
  citopatológicos.
- No SHAP da Regressão Logística, `texture_worst`, `concave_points_mean`, `radius_se` e `symmetry_worst` lideram.
  Algumas variáveis `_se` (`compactness_se`, `fractal_dimension_se`) têm efeito negativo, resultado da colinearidade:
  o modelo as usa para "corrigir" variáveis muito correlacionadas. Por isso interpretamos **grupos** de variáveis, não um coeficiente isolado.
- O gráfico *waterfall* (`14`) explica a previsão de **um paciente**, que é o que o médico veria no sistema.

## 7. Discussão crítica

**O modelo pode ser usado na prática?** Sim, como **ferramenta de triagem e segunda opinião**, não como diagnóstico final.

Limitações:
- Base pequena (569 casos), de uma única instituição e dos anos 90: é preciso validar com dados do próprio hospital.
- As variáveis dependem de um processo específico de digitalização e segmentação das imagens; mudanças de equipamento
  ou protocolo podem degradar o desempenho (*data drift*).
- Ainda houve um câncer não detectado; o teste tem só 114 casos, então as métricas têm incerteza considerável.

Proposta de uso:
1. O sistema calcula a probabilidade de malignidade e mostra as variáveis que mais influenciaram (SHAP).
2. Casos de maior risco são priorizados na fila do patologista.
3. O limiar é calibrado com a equipe clínica, privilegiando sensibilidade.
4. **O médico sempre tem a palavra final**; o modelo é monitorado, auditado e re-treinado periodicamente.

## 8. Próximos passos
- Validação cruzada repetida e intervalos de confiança das métricas.
- Validação externa com dados do hospital.
- EXTRA: CNN para detecção de pneumonia em radiografias (Chest X-Ray), integrando dados de imagem ao sistema.
