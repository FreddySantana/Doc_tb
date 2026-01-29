# Comparação: Modelos White Box vs Black Box

## Resumo Executivo

**Melhor Modelo (F1-Score):** LightGBM (Black Box) - F1=0.8611

**Melhor White Box:** Regressão Logística - F1=0.5625

**Melhor Black Box:** LightGBM - F1=0.8611

**Gap de Performance:** 0.2986 (53.09%)

## Tabela Comparativa

| Modelo              | Tipo      |   Accuracy |   Precision |   Recall |   F1-Score |    AUC |
|:--------------------|:----------|-----------:|------------:|---------:|-----------:|-------:|
| LightGBM            | Black Box |     0.9500 |      0.9688 |   0.7750 |     0.8611 | 0.9641 |
| CatBoost            | Black Box |     0.9450 |      0.9394 |   0.7750 |     0.8493 | 0.9730 |
| XGBoost             | Black Box |     0.9450 |      0.9677 |   0.7500 |     0.8451 | 0.9634 |
| Regressão Logística | White Box |     0.7900 |      0.4821 |   0.6750 |     0.5625 | 0.8697 |
| Árvore de Decisão   | White Box |     0.7250 |      0.3881 |   0.6500 |     0.4860 | 0.7931 |

## Análise

### Trade-off Interpretabilidade vs Performance

Os modelos **white box** (Regressão Logística e Árvore de Decisão) oferecem **máxima interpretabilidade**, permitindo que profissionais de saúde entendam exatamente quais fatores contribuem para a predição.

Os modelos **black box** (XGBoost, LightGBM, CatBoost) apresentam **melhor performance** (+0.2986 F1-Score), mas são menos interpretáveis diretamente.

### Recomendação

Para o contexto clínico de tuberculose, recomenda-se:

1. **Usar modelos black box** para predições (maior acurácia)
2. **Aplicar XAI (SHAP/LIME)** para explicar as predições
3. **Validar com modelos white box** para garantir coerência clínica

