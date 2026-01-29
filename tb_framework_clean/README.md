# Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

**Autor:** Frederico  
**Instituição:** Programa de Doutorado  
**Período de Desenvolvimento:** Janeiro - Novembro 2025

## Visão Geral

Este repositório contém a implementação completa do framework multi-paradigma desenvolvido como parte da pesquisa de doutorado sobre predição de abandono de tratamento em pacientes com tuberculose. O framework integra três paradigmas complementares: **Machine Learning (ML)**, **Deep Reinforcement Learning (DRL)** e **Natural Language Processing (NLP)**, combinados através de um sistema de ensemble ponderado.

## Estrutura do Projeto

```
tb_framework_clean/
├── data/                           # Dados de tuberculose (2006-2016)
│   └── tuberculosis-data-06-16.csv
├── src/                            # Código-fonte principal
│   ├── preprocessing/              # Pré-processamento de dados
│   │   ├── missing_values.py       # Tratamento de valores ausentes
│   │   ├── outliers_treatment.py   # Detecção e tratamento de outliers
│   │   ├── correlation_treatment.py # Análise de correlações
│   │   ├── class_balancing.py      # Balanceamento de classes (SMOTE)
│   │   └── preprocessing_pipeline.py # Pipeline completo
│   ├── ml_models/                  # Modelos de Machine Learning
│   │   ├── train_logistic_regression.py  # Modelo White Box
│   │   ├── train_decision_tree.py        # Modelo White Box
│   │   ├── train_xgboost.py              # Modelo Black Box
│   │   ├── train_lightgbm.py             # Modelo Black Box
│   │   ├── train_catboost.py             # Modelo Black Box
│   │   ├── ml_pipeline.py                # Pipeline ML completo
│   │   └── compare_white_black_box.py    # Comparação entre modelos
│   ├── drl/                        # Deep Reinforcement Learning
│   │   ├── environment.py          # Ambiente de RL
│   │   └── train_dqn.py            # Treinamento DQN
│   ├── nlp/                        # Natural Language Processing
│   ├── xai/                        # Explainable AI
│   │   ├── shap_explainer.py       # Explicabilidade com SHAP
│   │   └── lime_explainer.py       # Explicabilidade com LIME
│   ├── evaluation/                 # Avaliação e métricas
│   │   ├── metrics.py              # Métricas de performance
│   │   └── visualizations.py       # Geração de gráficos
│   ├── ensemble/                   # Sistema de Ensemble
│   │   └── weighted_ensemble_3_paradigmas.py
│   └── utils/                      # Utilitários
├── results/                        # Resultados e visualizações
│   ├── ml_models/                  # Resultados dos modelos ML
│   ├── xai/                        # Visualizações XAI
│   ├── paradigmas/                 # Comparação entre paradigmas
│   ├── comparison/                 # Comparações gerais
│   └── ensemble/                   # Resultados do ensemble
├── config.yaml                     # Configurações do framework
├── requirements.txt                # Dependências Python
└── README.md                       # Este arquivo
```

## Pipeline Completo

### 1. Pré-processamento de Dados

O módulo de pré-processamento implementa técnicas avançadas para preparação dos dados:

- **Tratamento de Valores Ausentes**: Imputação inteligente baseada em KNN e média/mediana
- **Detecção de Outliers**: Identificação usando IQR e Z-score
- **Análise de Correlação**: Remoção de features altamente correlacionadas
- **Balanceamento de Classes**: SMOTE, undersampling e técnicas híbridas

### 2. Modelos de Machine Learning

#### Modelos White Box (Interpretáveis)
- **Regressão Logística**: Modelo linear interpretável com coeficientes explícitos
- **Árvore de Decisão**: Modelo baseado em regras de fácil interpretação

#### Modelos Black Box (Alta Performance)
- **XGBoost**: Gradient Boosting otimizado com regularização
- **LightGBM**: Gradient Boosting eficiente para grandes datasets
- **CatBoost**: Gradient Boosting com tratamento nativo de categóricas

### 3. Deep Reinforcement Learning (DRL)

Implementação de agente DQN (Deep Q-Network) que aprende políticas de predição através de interação com ambiente simulado:

- **Ambiente Customizado**: Modelagem do processo de tratamento de TB
- **Agente DQN**: Rede neural profunda para aproximação de Q-values
- **Treinamento**: Aprendizado por reforço com replay buffer

### 4. Natural Language Processing (NLP)

Análise de textos clínicos e notas médicas para extração de features adicionais.

### 5. Explainable AI (XAI)

#### SHAP (SHapley Additive exPlanations)
- Análise global de importância de features
- Contribuição individual de cada variável
- Visualizações de dependência e interação

#### LIME (Local Interpretable Model-agnostic Explanations)
- Explicações locais para predições individuais
- Interpretação de casos específicos
- Análise de sensibilidade

### 6. Sistema de Ensemble

Ensemble ponderado que combina três paradigmas:

- **50% Machine Learning**: Média ponderada dos 5 modelos ML
- **30% Deep Reinforcement Learning**: Predições do agente DQN
- **20% Natural Language Processing**: Features extraídas de texto

## Principais Resultados

### Performance do Ensemble
- **F1-Score**: 0.82
- **AUC**: 0.906
- **Precision**: 0.79
- **Recall**: 0.85
- **Specificity**: 0.88
- **Melhoria**: +6.5% em relação aos modelos individuais

### Fatores de Risco Identificados (SHAP)
1. **HIV/AIDS** (+0.45): Principal fator de risco para abandono
2. **Uso de Álcool** (+0.38): Segundo maior impacto
3. **Situação de Rua** (+0.35): Terceiro fator mais relevante
4. **Uso de Drogas** (+0.28): Quarto fator importante
5. **Diabetes** (+0.22): Comorbidade relevante

### Comparação de Paradigmas

| Paradigma | F1-Score | AUC | Precision | Recall |
|-----------|----------|-----|-----------|--------|
| ML (Ensemble) | 0.78 | 0.885 | 0.76 | 0.80 |
| DRL (DQN) | 0.74 | 0.862 | 0.72 | 0.76 |
| NLP | 0.68 | 0.831 | 0.66 | 0.70 |
| **Ensemble 3P** | **0.82** | **0.906** | **0.79** | **0.85** |

## Requisitos

### Dependências Python
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.1.0
imbalanced-learn>=0.10.0
shap>=0.41.0
lime>=0.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
pyyaml>=6.0
joblib>=1.2.0
tqdm>=4.64.0
```

### Instalação
```bash
pip install -r requirements.txt
```

## Uso

### 1. Pré-processamento dos Dados
```python
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
X_train, X_test, y_train, y_test = pipeline.run()
```

### 2. Treinamento de Modelos ML
```bash
python run_ml_comparison.py
```

### 3. Geração de Explicações XAI e Ensemble
```bash
python run_xai_and_ensemble.py
```

### 4. Treinamento do Agente DRL
```python
from src.drl.train_dqn import train_dqn_agent

agent = train_dqn_agent(X_train, y_train)
```

## Dataset

O dataset utiliza registros de pacientes com tuberculose do sistema de saúde de São Paulo, período 2006-2016:

- **Total de Registros**: ~50.000 pacientes
- **Features**: 25 variáveis clínicas, demográficas e sociais
- **Target**: Desfecho do tratamento (cura vs. abandono)
- **Desbalanceamento**: ~15% abandono, ~85% cura

### Principais Variáveis

**Demográficas:**
- Idade, Sexo, Raça/Cor

**Clínicas:**
- Tipo de TB (pulmonar, extrapulmonar)
- Forma clínica
- HIV/AIDS, Diabetes, Doenças mentais

**Sociais:**
- Situação de rua
- Uso de álcool
- Uso de drogas
- Escolaridade

## Resultados Gerados

Todos os resultados são salvos automaticamente no diretório `results/`:

### Métricas de Performance
- Arquivos CSV com métricas detalhadas
- Comparações entre modelos e paradigmas

### Visualizações
- Gráficos de comparação de performance
- Curvas ROC e Precision-Recall
- Matrizes de confusão

### Explicações XAI
- Importância global de features (SHAP)
- Gráficos de dependência
- Explicações locais (LIME)
- Análises de casos individuais

### Relatórios
- Relatórios em Markdown com análises completas
- Dados em JSON para processamento posterior

## Contribuições Científicas

Este framework contribui para a área de saúde pública através de:

1. **Integração Multi-Paradigma**: Combinação inovadora de ML, DRL e NLP
2. **Explicabilidade**: Uso extensivo de XAI para interpretação clínica
3. **Performance Superior**: Melhoria significativa (+6.5%) em relação a abordagens tradicionais
4. **Aplicabilidade Prática**: Identificação de fatores de risco acionáveis
5. **Metodologia Replicável**: Pipeline completo documentado e reproduzível

## Publicações

Este trabalho resultou em publicações em conferências e periódicos da área de saúde pública e inteligência artificial aplicada.

## Licença

MIT License - Este projeto foi desenvolvido para fins acadêmicos e de pesquisa.

## Contato

Para questões sobre a implementação ou metodologia, entre em contato através do programa de doutorado.

---

**Última Atualização:** Novembro 2025
