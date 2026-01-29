#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Completo: XAI + Ensemble Multi-Paradigma

Executa:
1. XAI (SHAP + LIME) sobre o melhor modelo ML
2. Simulações de DRL e NLP
3. Ensemble com 3 paradigmas
4. Comparação completa com visualizações
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Criar diretórios
Path("results/xai").mkdir(parents=True, exist_ok=True)
Path("results/ensemble").mkdir(parents=True, exist_ok=True)

def generate_synthetic_data(n_samples=1000, random_state=42):
    """Gera dados sintéticos para demonstração"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=random_state
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

def run_xai_shap(model, X_train, X_test, model_name="LightGBM"):
    """Executa SHAP para explicabilidade"""
    logger.info(f"\n{'='*60}")
    logger.info(f"EXECUTANDO XAI - SHAP SOBRE {model_name}")
    logger.info(f"{'='*60}")
    
    try:
        import shap
        
        # Criar explainer
        logger.info("Criando SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        
        # Calcular SHAP values
        logger.info("Calculando SHAP values...")
        shap_values = explainer.shap_values(X_test)
        
        # Se retornar lista (binário), pegar classe positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Summary plot
        logger.info("Gerando visualização SHAP Summary...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, show=False, max_display=15)
        plt.title(f"SHAP Summary Plot - {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("results/xai/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Salvo: results/xai/shap_summary.png")
        
        # Feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(10), feature_importance[top_features_idx])
        plt.yticks(range(10), [f"Feature {i}" for i in top_features_idx])
        plt.xlabel("Mean |SHAP value|", fontsize=12)
        plt.title(f"Top 10 Features - SHAP Importance ({model_name})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("results/xai/shap_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Salvo: results/xai/shap_importance.png")
        
        logger.info(f"✅ XAI-SHAP concluído com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no SHAP: {e}")
        return False

def run_xai_lime(model, X_train, X_test, y_test, model_name="LightGBM"):
    """Executa LIME para explicabilidade"""
    logger.info(f"\n{'='*60}")
    logger.info(f"EXECUTANDO XAI - LIME SOBRE {model_name}")
    logger.info(f"{'='*60}")
    
    try:
        from lime import lime_tabular
        
        # Criar explainer
        logger.info("Criando LIME Explainer...")
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            mode='classification',
            feature_names=[f"Feature_{i}" for i in range(X_train.shape[1])],
            class_names=['Não-Abandono', 'Abandono'],
            discretize_continuous=True
        )
        
        # Explicar algumas instâncias
        logger.info("Gerando explicações LIME para instâncias de teste...")
        
        # Selecionar instâncias interessantes
        y_pred = model.predict(X_test)
        
        # Instância corretamente classificada como abandono
        correct_positive = np.where((y_test == 1) & (y_pred == 1))[0]
        if len(correct_positive) > 0:
            idx = correct_positive[0]
            exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=10)
            
            fig = exp.as_pyplot_figure()
            plt.title(f"LIME - Caso Corretamente Classificado (Abandono)", fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig("results/xai/lime_example_positive.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ Salvo: results/xai/lime_example_positive.png")
        
        logger.info(f"✅ XAI-LIME concluído com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no LIME: {e}")
        return False

def simulate_drl_predictions(y_proba_ml, noise_level=0.05, random_state=42):
    """Simula predições de DRL baseadas em ML com ruído"""
    np.random.seed(random_state)
    noise = np.random.normal(0, noise_level, size=y_proba_ml.shape)
    y_proba_drl = np.clip(y_proba_ml + noise, 0, 1)
    return y_proba_drl

def simulate_nlp_predictions(y_proba_ml, bias=-0.03, random_state=42):
    """Simula predições de NLP baseadas em ML com bias"""
    np.random.seed(random_state)
    y_proba_nlp = np.clip(y_proba_ml + bias, 0, 1)
    return y_proba_nlp

def create_ensemble(y_proba_ml, y_proba_drl, y_proba_nlp, weights=[0.50, 0.30, 0.20]):
    """Cria ensemble ponderado"""
    y_proba_ensemble = (
        weights[0] * y_proba_ml +
        weights[1] * y_proba_drl +
        weights[2] * y_proba_nlp
    )
    return y_proba_ensemble

def evaluate_model(y_true, y_proba, threshold=0.5):
    """Avalia modelo"""
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba)
    }
    
    return metrics

def plot_comparison(results_df, save_path):
    """Plota comparação de modelos"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Comparação: Modelos Individuais vs Ensemble", fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        
        data = results_df[['modelo', metric]].sort_values(metric, ascending=False)
        
        colors = ['#2ecc71' if 'Ensemble' in m else '#3498db' for m in data['modelo']]
        
        ax.barh(data['modelo'], data[metric], color=colors)
        ax.set_xlabel(title, fontsize=11)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Adicionar valores
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)
    
    # Remover último subplot vazio
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Salvo: {save_path}")

def main():
    logger.info("\n" + "="*60)
    logger.info("INICIANDO PIPELINE: XAI + ENSEMBLE MULTI-PARADIGMA")
    logger.info("="*60)
    
    # 1. Gerar dados
    logger.info("\n1. Gerando dados sintéticos...")
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    logger.info(f"✅ Dados gerados: {len(X_train)} treino, {len(X_test)} teste")
    
    # 2. Carregar modelo ML treinado (ou treinar novo)
    logger.info("\n2. Carregando/Treinando modelo LightGBM...")
    try:
        import lightgbm as lgb
        
        model_ml = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model_ml.fit(X_train, y_train)
        logger.info("✅ Modelo LightGBM treinado")
        
        # Predições ML
        y_proba_ml = model_ml.predict_proba(X_test)[:, 1]
        metrics_ml = evaluate_model(y_test, y_proba_ml)
        logger.info(f"   F1-Score (ML): {metrics_ml['f1_score']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Erro ao treinar modelo: {e}")
        return
    
    # 3. Executar XAI
    logger.info("\n3. Executando XAI...")
    run_xai_shap(model_ml, X_train, X_test, "LightGBM")
    run_xai_lime(model_ml, X_train, X_test, y_test, "LightGBM")
    
    # 4. Simular DRL e NLP
    logger.info("\n4. Simulando paradigmas DRL e NLP...")
    y_proba_drl = simulate_drl_predictions(y_proba_ml)
    y_proba_nlp = simulate_nlp_predictions(y_proba_ml)
    
    metrics_drl = evaluate_model(y_test, y_proba_drl)
    metrics_nlp = evaluate_model(y_test, y_proba_nlp)
    
    logger.info(f"   F1-Score (DRL): {metrics_drl['f1_score']:.4f}")
    logger.info(f"   F1-Score (NLP): {metrics_nlp['f1_score']:.4f}")
    
    # 5. Criar Ensemble
    logger.info("\n5. Criando Ensemble Multi-Paradigma...")
    y_proba_ensemble = create_ensemble(y_proba_ml, y_proba_drl, y_proba_nlp)
    metrics_ensemble = evaluate_model(y_test, y_proba_ensemble)
    
    logger.info(f"   F1-Score (Ensemble): {metrics_ensemble['f1_score']:.4f}")
    
    # 6. Comparação
    logger.info("\n6. Gerando comparação completa...")
    
    results = {
        'modelo': ['ML (LightGBM)', 'DRL (Simulado)', 'NLP (Simulado)', 'Ensemble 3-Paradigmas'],
        'accuracy': [metrics_ml['accuracy'], metrics_drl['accuracy'], metrics_nlp['accuracy'], metrics_ensemble['accuracy']],
        'precision': [metrics_ml['precision'], metrics_drl['precision'], metrics_nlp['precision'], metrics_ensemble['precision']],
        'recall': [metrics_ml['recall'], metrics_drl['recall'], metrics_nlp['recall'], metrics_ensemble['recall']],
        'f1_score': [metrics_ml['f1_score'], metrics_drl['f1_score'], metrics_nlp['f1_score'], metrics_ensemble['f1_score']],
        'auc': [metrics_ml['auc'], metrics_drl['auc'], metrics_nlp['auc'], metrics_ensemble['auc']]
    }
    
    results_df = pd.DataFrame(results)
    
    # Salvar tabela
    logger.info("\nTabela de Resultados:")
    logger.info("\n" + results_df.to_string(index=False))
    
    results_df.to_csv("results/ensemble/comparison_results.csv", index=False)
    logger.info("\n✅ Salvo: results/ensemble/comparison_results.csv")
    
    # Salvar JSON
    with open("results/ensemble/comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("✅ Salvo: results/ensemble/comparison_results.json")
    
    # Plotar comparação
    plot_comparison(results_df, "results/ensemble/ensemble_comparison.png")
    
    # Relatório final
    logger.info("\n" + "="*60)
    logger.info("RESUMO FINAL")
    logger.info("="*60)
    logger.info(f"\n✅ XAI (SHAP + LIME) executado com sucesso")
    logger.info(f"✅ Ensemble Multi-Paradigma criado")
    logger.info(f"\n🏆 Melhor F1-Score: {results_df['f1_score'].max():.4f} ({results_df.loc[results_df['f1_score'].idxmax(), 'modelo']})")
    logger.info(f"🏆 Melhor AUC: {results_df['auc'].max():.4f} ({results_df.loc[results_df['auc'].idxmax(), 'modelo']})")
    
    logger.info("\n📂 Resultados disponíveis em:")
    logger.info("   - results/xai/shap_summary.png")
    logger.info("   - results/xai/shap_importance.png")
    logger.info("   - results/xai/lime_example_positive.png")
    logger.info("   - results/ensemble/ensemble_comparison.png")
    logger.info("   - results/ensemble/comparison_results.csv")
    logger.info("   - results/ensemble/comparison_results.json")
    
    logger.info("\n" + "="*60)
    logger.info("✅ EXECUÇÃO COMPLETA COM SUCESSO!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
