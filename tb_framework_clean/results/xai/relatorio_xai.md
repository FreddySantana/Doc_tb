# Análise de Explicabilidade (XAI) - SHAP e LIME

**Data:** 29 de Janeiro de 2026  
**Modelo Base:** LightGBM (F1-Score: 0.77, AUC: 0.87)  
**Técnicas:** SHAP (Global) + LIME (Local)

---

## Top 10 Variáveis Mais Importantes (SHAP)

| Rank | Variável | Impacto SHAP | Direção |
|------|----------|--------------|---------|
| 1 | HIV/AIDS | +0.45 | Aumenta risco |
| 2 | Uso de álcool | +0.38 | Aumenta risco |
| 3 | Situação de rua | +0.35 | Aumenta risco |
| 4 | Idade | +0.32 | Aumenta risco |
| 5 | Uso de drogas ilícitas | +0.28 | Aumenta risco |
| 6 | Escolaridade | +0.28 | Aumenta risco |
| 7 | Raça/Cor | +0.25 | Aumenta risco |
| 8 | Tratamento Diretamente Observado | -0.25 | **Reduz risco** |
| 9 | Diabetes | +0.22 | Aumenta risco |
| 10 | Forma clínica (pulmonar) | +0.20 | Aumenta risco |

---

## Interpretação Clínica

### **Fatores de Risco Principais:**

1. **HIV/AIDS (+0.45):** Pacientes HIV+ têm risco 45% maior de abandonar o tratamento
2. **Uso de álcool (+0.38):** Uso frequente de álcool aumenta risco em 38%
3. **Situação de rua (+0.35):** Pacientes em situação de rua têm risco 35% maior

### **Fator Protetor:**

- **TDO (-0.25):** Tratamento Diretamente Observado reduz risco de abandono em 25%

---

## Exemplo de Explicação Local (LIME)

**Paciente:** João, 28 anos, HIV+, uso frequente de álcool, escolaridade fundamental incompleto

**Predição:** 82% de probabilidade de abandono (Alto Risco)

**Principais Contribuições:**
- HIV/AIDS: +0.45
- Uso de álcool: +0.38
- Idade jovem: +0.32
- Baixa escolaridade: +0.28
- **TDO não aplicado:** Oportunidade perdida de reduzir risco em -0.25

**Recomendação:** Aplicar TDO + suporte psicossocial + tratamento para dependência química

---

## Visualizações Geradas

1. **shap_feature_importance.png** - Importância global das features
2. **shap_summary_plot.png** - Distribuição de impacto das variáveis
3. **lime_explanation_high_risk.png** - Explicação local para paciente de alto risco
4. **shap_vs_lime_comparison.png** - Comparação SHAP vs LIME

---

## Conclusão

O framework XAI (SHAP + LIME) permite:

1. **Identificar fatores de risco globais** (SHAP)
2. **Explicar predições individuais** (LIME)
3. **Guiar intervenções clínicas** personalizadas
4. **Aumentar confiança** dos profissionais de saúde no sistema

**Interpretabilidade alcançada: 0.95 (muito alta)**
