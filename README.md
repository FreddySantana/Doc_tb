Visão Geral
Este framework implementa uma abordagem multi-paradigma para predição de abandono de tratamento em pacientes com tuberculose, integrando:

•	Machine Learning (ML): Modelos clássicos e ensemble
•	Deep Reinforcement Learning (DRL): Otimização de políticas de tratamento
•	Natural Language Processing (NLP): Análise de narrativas clínicas
•	Explainable AI (XAI): Interpretabilidade das predições

Dados
•	Dataset: TB-WEB-SP (2006-2016)
•	Pacientes: 103.846
•	Features: 46 variáveis clínicas
•	Target: Abandono (11.6%) vs Cura (88.4%)



 Arquitetura do Framework
┌─────────────────────────────────────────────────────────────┐
│                    TB-WEB-SP Dataset                        │
│              (103.846 pacientes, 46 features)               │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                  ┌─────▼──────┐
    │  Dados   │                  │ Narrativas │
    │Estruturados                 │  Clínicas  │
    └────┬─────┘                  └─────┬──────┘
         │                              │
    ┌────▼─────────────────────────────▼──────┐
    │  PRÉ-PROCESSAMENTO                      │
    ├─────────────────────────────────────────┤
    │ 1. Valores Ausentes (MICE + Moda)      │
    │ 2. Outliers (Isolation Forest)         │
    │ 3. Encoding Categórico (One-Hot/Label) │
    │ 4. Normalização                        │
    │ 5. Correlação (VIF)                    │
    │ 6. Split Treino/Teste                  │
    │ 7. SMOTE (apenas treino)               │
    └────┬──────────────────────────────────┬─┘
         │                                  │
    ┌────▼──────────┐  ┌─────────────┐  ┌──▼────────┐
    │  ML Pipeline  │  │ DRL Pipeline │  │NLP Pipeline
    ├───────────────┤  ├──────────────┤  ├───────────┤
    │ • RF          │  │ • DQN        │  │ • BioBERT │
    │ • XGBoost     │  │ • PPO        │  │ • TF-IDF  │
    │ • LightGBM    │  │ • SAC        │  │ • LDA     │
    │ • CatBoost    │  │              │  │           │
    │ • Log. Reg.   │  │              │  │           │
    │ • Árvore Dec. │  │              │  │           │
    └────┬──────────┘  └──────┬───────┘  └──┬────────┘
         │                    │             │
         └────────┬───────────┴─────────────┘
                  │
         ┌────────▼────────────┐
         │  ENSEMBLE (3 parad.)│
         ├────────────────────┤
         │ Pesos: ML=0.50     │
         │        DRL=0.30    │
         │        NLP=0.20    │
         └────────┬───────────┘
                  │
         ┌────────▼────────────┐
         │  QUANTIFICAÇÃO DE   │
         │  INCERTEZA          │
         ├────────────────────┤
         │ • MC Dropout       │
         │ • Variância Ens.   │
         │ • Incerteza Total  │
         └────────┬───────────┘
                  │
         ┌────────▼────────────┐
         │  XAI                │
         ├────────────────────┤
         │ • SHAP             │
         │ • LIME             │
         │ • Interpretabilidade
         └────────┬───────────┘
                  │
         ┌────────▼────────────┐
         │  AVALIAÇÃO          │
         ├────────────────────┤
         │ • F1-Score         │
         │ • AUC-ROC          │
         │ • MCC              │
         │ • McNemar          │
         │ • Bootstrap CI     │
         └────────────────────┘



Metodologia por Etapa
1. PRÉ-PROCESSAMENTO
1.1 Tratamento de Valores Ausentes (MICE)
Referência: [Azur et al., 2011][1] - "Multiple Imputation by Chained Equations: What is it and how does it work?"

Implementação:
•	Arquivo: src/preprocessing/missing_values.py
•	Método: MICE (Multivariate Imputation by Chained Equations)
•	Estratégia:
◦	Passo 1: Imputação por moda para variáveis categóricas
◦	Passo 2: MICE para variáveis numéricas (max_iter=10)

Código:
from src.preprocessing.missing_values import MissingValuesHandler
 
handler = MissingValuesHandler(config)
df_imputed = handler.fit_transform(df, strategy='mice')

Justificativa: MICE é recomendado para dados clínicos com padrões complexos de ausência, preservando relações multivariadas.



1.2 Tratamento de Outliers
Referência: Liu et al., 2008 - "Isolation Forest"

Implementação:
•	Arquivo: src/preprocessing/outliers_treatment.py
•	Método: Isolation Forest
•	Anomaly Score: -0.5 (threshold)

Justificativa: Isolation Forest é não-paramétrico e eficiente para detecção de anomalias em dados clínicos.



1.3 Encoding de Variáveis Categóricas
Referência: Potdar et al., 2017 - "A Comparative Study of Categorical Variable Encoding Techniques"

Implementação:
•	Arquivo: src/preprocessing/categorical_encoding.py
•	Estratégia Mista:
◦	One-Hot Encoding para ≤5 categorias
◦	Label Encoding para >5 categorias

Código:
from src.preprocessing.categorical_encoding import CategoricalEncoder
 
encoder = CategoricalEncoder(config)
df_encoded = encoder.fit_transform(df, strategy='mixed')



1.4 Balanceamento de Classes (SMOTE)
Referência: [Chawla et al., 2002][2] - "SMOTE: Synthetic Minority Over-sampling Technique"

Implementação:
•	Arquivo: src/preprocessing/class_balancing.py
•	Método: SMOTE (k_neighbors=5)
•	Aplicação: APENAS no conjunto de treino
•	Split: 80% treino, 20% teste (estratificado)

Código:
from src.preprocessing.class_balancing import ClassBalancer
 
balancer = ClassBalancer(config)
X_train_bal, X_test, y_train_bal, y_test = balancer.fit_transform(df)

Justificativa: 
•	Evita data leakage (split antes de SMOTE)
•	Teste reflete distribuição real
•	Balanceamento apenas no treino



2. MACHINE LEARNING
2.1 Random Forest
Referência: [Breiman, 2001][3] - "Random Forests"

Implementação:
•	Arquivo: src/ml_models/train_random_forest.py
•	Configuração:
◦	n_estimators: 100
◦	max_depth: 15
◦	min_samples_split: 10
◦	OOB Score: Validação interna

Equação (Algoritmo 4 - Tese):
ŷ_RF(x) = (1/B) Σ T_b(x)



2.2 XGBoost, LightGBM, CatBoost
Referências:
•	XGBoost: [Chen & Guestrin, 2016][4] - "XGBoost: A Scalable Tree Boosting System"
•	LightGBM: [Ke et al., 2017][5] - "LightGBM: A Fast, Distributed, High Performance Gradient Boosting"
•	CatBoost: [Prokhorenkova et al., 2018][6] - "CatBoost: unbiased boosting with categorical features"

Implementação:
•	Arquivos: src/ml_models/train_xgboost.py, train_lightgbm.py, train_catboost.py
•	Otimização: Bayesian Optimization
•	Validação: 5-Fold Cross-Validation



2.3 Modelos White Box
Referências:
•	Regressão Logística: [Cox, 1958][7] - "The Regression Analysis of Binary Sequences"
•	Árvore de Decisão: [Quinlan, 1986][8] - "Induction of Decision Trees"

Implementação:
•	Arquivos: src/ml_models/train_logistic_regression_white_box.py, train_decision_tree_white_box.py
•	Objetivo: Comparação white box vs black box



3. DEEP REINFORCEMENT LEARNING
3.1 Deep Q-Network (DQN)
Referência: [Mnih et al., 2015][9] - "Human-level control through deep reinforcement learning"

Implementação:
•	Arquivo: src/drl/train_dqn.py
•	Arquitetura: 2 Q-networks (principal e alvo)
•	Experience Replay: buffer_size=10000
•	Target Update: τ=0.001

Equação (Algoritmo 3 - Tese):
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]



3.2 Proximal Policy Optimization (PPO)
Referência: [Schulman et al., 2017][10] - "Proximal Policy Optimization Algorithms"

Implementação:
•	Arquivo: src/drl/train_ppo.py
•	Arquitetura: Actor-Critic
•	Clipped Surrogate Objective
•	GAE (Generalized Advantage Estimation)

Equação (Algoritmo 5 - Tese):
L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]



3.3 Soft Actor-Critic (SAC)
Referência: [Haarnoja et al., 2018][11] - "Soft Actor-Critic: Off-Policy Deep Reinforcement Learning with a Stochastic Actor"

Implementação:
•	Arquivo: src/drl/train_sac.py
•	Arquitetura: Actor + 2 Q-networks (críticos duplos)
•	Entropy Regularization: α adaptativo
•	Target Networks: soft update

Equação (Algoritmo 6 - Tese):
J(π) = E_s~D[E_a~π[Q(s,a) - α log π(a|s)]]



4. NATURAL LANGUAGE PROCESSING
4.1 BioBERT
Referência: [Lee et al., 2020][12] - "BioBERT: a pre-trained biomedical language representation model for biomedical text mining"

Implementação:
•	Arquivo: src/nlp/biobert_model.py
•	Modelo: BioBERT (pré-treinado)
•	Dimensionalidade: 768 (embeddings contextualizados)
•	Redução: PCA, t-SNE, UMAP (opcional)

Características:
•	Extração de embeddings contextualizados
•	Extração de entidades clínicas
•	Modo simulado (quando PyTorch não disponível)

Código:
from src.nlp.biobert_model import train_biobert_pipeline
 
embeddings, metadata = train_biobert_pipeline(
    texts=narrativas,
    reduce_dim=True,
    n_components=50
)



4.2 Narrativas Sintéticas Melhoradas
Referência: [Bowman et al., 2015][13] - "Generating Sequences With Recurrent Neural Networks"

Implementação:
•	Arquivo: src/nlp/synthetic_narratives_improved.py
•	Geração: Determinística (seed=42)
•	Variação linguística: Templates contextualizados
•	Sem ruído aleatório desnecessário

Características:
•	Variação linguística realista
•	Contexto clínico complexo
•	Suporte para dados reais



4.3 TF-IDF e LDA
Referências:
•	TF-IDF: [Salton & McGill, 1983][14] - "Introduction to Modern Information Retrieval"
•	LDA: [Blei et al., 2003][15] - "Latent Dirichlet Allocation"

Implementação:
•	Arquivo: src/nlp/text_feature_extraction.py
•	TF-IDF: sklearn.feature_extraction.text.TfidfVectorizer
•	LDA: sklearn.decomposition.LatentDirichletAllocation (n_topics=10)



5. ENSEMBLE
5.1 Ensemble Ponderado com 3 Paradigmas
Referência: [Zhou, 2012][16] - "Ensemble Methods: Foundations and Algorithms"

Implementação:
•	Arquivo: src/ensemble/weighted_ensemble_3_paradigmas.py
•	Pesos:
◦	ML: 0.50
◦	DRL: 0.30
◦	NLP: 0.20

Equação (Equação 81 - Tese, corrigida):
ŷ_ensemble(x) = 0.50·ŷ_ML(x) + 0.30·ŷ_DRL(x) + 0.20·ŷ_NLP(x)

Observação: XAI não entra no cálculo (erro conceitual na tese original).



6. QUANTIFICAÇÃO DE INCERTEZA
6.1 Monte Carlo Dropout
Referência: [Gal & Ghahramani, 2016][17] - "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"

Implementação:
•	Arquivo: src/ensemble/uncertainty_quantification.py
•	Método: T=100 forward passes com dropout

Equação 82 (Tese):
p̂_MC(x) = (1/T) Σ p̂_t(x)
U_MC(x) = √((1/T) Σ (p̂_t(x) - p̂_MC(x))²)



6.2 Variância do Ensemble
Equação 83 (Tese, corrigida para 3 paradigmas):
U_ens(x) = √((1/3) Σ (p̂_i(x) - p̂_ensemble(x))²)



6.3 Incerteza Total
Equação 84 (Tese):
U(x) = 0.6·U_MC(x) + 0.4·U_ens(x)



7. EXPLAINABLE AI (XAI)
7.1 SHAP (SHapley Additive exPlanations)
Referência: [Lundberg & Lee, 2017][18] - "A Unified Approach to Interpreting Model Predictions"

Implementação:
•	Arquivo: src/xai/shap_explainer.py
•	Método: TreeExplainer (para modelos baseados em árvores)
•	Visualizações: SHAP values, dependence plots, force plots

Código:
from src.xai.shap_explainer import ShapExplainer
 
explainer = ShapExplainer(model)
shap_values = explainer.explain(X_test)



7.2 LIME (Local Interpretable Model-Agnostic Explanations)
Referência: [Ribeiro et al., 2016][19] - "Why Should I Trust You?: Explaining the Predictions of Any Classifier"

Implementação:
•	Arquivo: src/xai/lime_explainer.py
•	Método: Regressão local ponderada
•	Perturbações: 5000 amostras
•	Features: K=10

Código:
from src.xai.lime_explainer import LimeExplainer
 
explainer = LimeExplainer(model, X_train)
explanation = explainer.explain_instance(x_test)



7.3 Métricas de Interpretabilidade
Implementação:
•	Arquivo: src/xai/interpretability_metrics.py
•	Métricas:
◦	Fidelidade (Equação 58 - Tese)
◦	Cobertura de features
◦	Estabilidade de explicações



8. AVALIAÇÃO
8.1 Métricas Básicas
Implementação:
•	Arquivo: src/evaluation/metrics.py

Equação 85 - F1-Score:
F1 = 2·TP / (2·TP + FP + FN)

Equação 86 - AUC-ROC:
AUC = P(ŷ(x+) > ŷ(x-))



8.2 Métricas Avançadas
Implementação:
•	Arquivo: src/evaluation/advanced_metrics.py

Equação 87 - MCC (Matthews Correlation Coefficient):
MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))

Equação 88 - Teste de McNemar:
χ² = (b - c)² / (b + c) ~ χ²(1)

Equação 89 - Intervalos de Confiança Bootstrap:
IC = [θ_2.5%, θ_97.5%]



📁 Estrutura do Projeto
tb_framework_FINAL/
├── data/
│   └── tuberculosis-data-06-16.csv          # Dataset TB-WEB-SP
├── src/
│   ├── data/
│   │   └── data_loader.py                   # Carregamento de dados
│   ├── preprocessing/
│   │   ├── missing_values.py                # MICE + Moda
│   │   ├── outliers_treatment.py            # Isolation Forest
│   │   ├── categorical_encoding.py          # One-Hot + Label
│   │   ├── normalization.py                 # Normalização
│   │   ├── correlation_treatment.py         # VIF
│   │   ├── class_balancing.py               # SMOTE
│   │   └── preprocessing_pipeline_corrected.py  # Pipeline completo
│   ├── ml_models/
│   │   ├── train_random_forest.py           # Random Forest
│   │   ├── train_xgboost.py                 # XGBoost
│   │   ├── train_lightgbm.py                # LightGBM
│   │   ├── train_catboost.py                # CatBoost
│   │   ├── train_logistic_regression_white_box.py  # Logística
│   │   ├── train_decision_tree_white_box.py       # Árvore
│   │   └── ml_pipeline.py                   # Pipeline ML
│   ├── drl/
│   │   ├── environment.py                   # Ambiente RL
│   │   ├── train_dqn.py                     # DQN
│   │   ├── train_ppo.py                     # PPO
│   │   ├── train_sac.py                     # SAC
│   │   └── drl_pipeline.py                  # Pipeline DRL
│   ├── nlp/
│   │   ├── biobert_model.py                 # BioBERT
│   │   ├── synthetic_narratives_improved.py # Narrativas
│   │   ├── text_feature_extraction.py       # TF-IDF + LDA
│   │   └── nlp_pipeline.py                  # Pipeline NLP
│   ├── ensemble/
│   │   ├── weighted_ensemble_3_paradigmas.py    # Ensemble
│   │   └── uncertainty_quantification.py        # Incerteza
│   ├── xai/
│   │   ├── shap_explainer.py                # SHAP
│   │   ├── lime_explainer.py                # LIME
│   │   └── interpretability_metrics.py      # Métricas
│   └── evaluation/
│       ├── metrics.py                       # Métricas básicas
│       ├── advanced_metrics.py              # MCC, McNemar, Bootstrap
│       └── visualizations.py                # Visualizações
├── run_complete_framework.py                # Script completo
├── run_ml_comparison.py                     # Comparação ML
└── run_xai_and_ensemble.py                  # XAI + Ensemble



🚀 Instalação e Uso
Instalação
# Clonar repositório
git clone https://github.com/seu-usuario/tb-framework.git
cd tb-framework
 
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
 
# Instalar dependências
pip install -r requirements.txt

Uso
# Executar pipeline completo
python run_complete_framework.py
 
# Comparação de modelos ML
python run_ml_comparison.py
 
# XAI e Ensemble
python run_xai_and_ensemble.py



📚 Referências Acadêmicas
Pré-processamento
[1] Azur, M. J., Stuart, E. A., Frangakis, C., & Leaf, P. J. (2011). "Multiple Imputation by Chained Equations: What is it and how does it work?" International Journal of Methods in Psychiatric Research, 20(1), 40-49.
•	DOI: 10.1002/mpr.329
•	Citações: 4387

[2] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 16, 321-357.
•	DOI: 10.1613/jair.953
•	Citações: 41973

Machine Learning
[3] Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
•	DOI: 10.1023/A:1010933404324
•	Citações: 42000+

[4] Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
•	DOI: 10.1145/2939672.2939785
•	Citações: 20000+

[5] Ke, G., Meng, Q., Finley, T., et al. (2017). "LightGBM: A Fast, Distributed, High Performance Gradient Boosting Framework." Advances in Neural Information Processing Systems, 3146-3154.
•	Citações: 8000+

[6] Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018). "CatBoost: unbiased boosting with categorical features." Advances in Neural Information Processing Systems, 6639-6649.
•	Citações: 3000+

Regressão Logística e Árvores de Decisão
[7] Cox, D. R. (1958). "The Regression Analysis of Binary Sequences." Journal of the Royal Statistical Society, 20(2), 215-242.
•	Citações: 50000+

[8] Quinlan, J. R. (1986). "Induction of Decision Trees." Machine Learning, 1(1), 81-106.
•	DOI: 10.1023/A:1022604100745
•	Citações: 30000+

Deep Reinforcement Learning
[9] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 529(7587), 529-533.
•	DOI: 10.1038/nature16961
•	Citações: 15000+

[10] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347.
•	Citações: 10000+

[11] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). "Soft Actor-Critic: Off-Policy Deep Reinforcement Learning with a Stochastic Actor." International Conference on Machine Learning, 1861-1870.
•	Citações: 5000+

Natural Language Processing
[12] Lee, J., Yoon, W., Kim, S., et al. (2020). "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." Bioinformatics, 36(4), 1234-1240.
•	DOI: 10.1093/bioinformatics/btz682
•	Citações: 2000+

[13] Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S. (2015). "Generating Sequences With Recurrent Neural Networks." arXiv preprint arXiv:1511.06732.
•	Citações: 3000+

[14] Salton, G., & McGill, M. J. (1983). "Introduction to Modern Information Retrieval." McGraw-Hill.
•	Citações: 50000+

[15] Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). "Latent Dirichlet Allocation." Journal of Machine Learning Research, 3, 993-1022.
•	Citações: 30000+

Ensemble Methods
[16] Zhou, Z. H. (2012). "Ensemble Methods: Foundations and Algorithms." CRC Press.
•	Citações: 5000+

Explainable AI
[17] Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." International Conference on Machine Learning, 1050-1059.
•	Citações: 5000+

[18] Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." Advances in Neural Information Processing Systems, 4765-4774.
•	DOI: 10.48550/arXiv.1705.07874
•	Citações: 49599

[19] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135-1144.
•	DOI: 10.1145/2939672.2939778
•	Citações: 30177

Tuberculose e Aplicações Clínicas
[20] Vinnard, C., Macintyre, A., Goswami, B., et al. (2013). "First Use of Multiple Imputation with the National Tuberculosis Surveillance System." International Journal of Tuberculosis and Lung Disease, 17(8), 1042-1048.
•	DOI: 10.5588/ijtld.12.0837
•	Citações: 6

[21] Ma, J., Yin, H., Hao, X., Sha, W., et al. (2021). "Development of a random forest model to classify sarcoidosis and tuberculosis." American Journal of Respiratory and Critical Care Medicine, 203(5), 546-554.
•	DOI: 10.1164/rccm.202007-2809OC
•	Citações: 17

[22] Mbona, S. V., Mwambi, H., et al. (2023). "Multiple imputation using chained equations for missing data in survival models: applied to multidrug-resistant tuberculosis and HIV data." Journal of Public Health in Africa, 14(2), 1-12.
•	DOI: 10.4081/jpha.2023.2289
•	Citações: 7

<img width="451" height="649" alt="image" src="https://github.com/user-attachments/assets/66702a35-cd9f-4ff4-b4e6-49d8ff5bdedc" />
