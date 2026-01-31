#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Comparação de Modelos ML
Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose
Data de Criação: 2024-07-01
Última Modificação: 2025-01-20
"""
import sys
sys.path.insert(0, '/home/ubuntu/tb_framework_phd')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import yaml

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carrega arquivo de configuração."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_synthetic_data(config: dict):
    """Cria dados sintéticos para teste."""
    logger.info("="*70)
    logger.info("CRIANDO DADOS SINTÉTICOS")
    logger.info("="*70)
    
    # Gerar dados
    X, y = make_classification(
        n_samples=1000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2],  # Desbalanceado como TB
        random_state=config['reproducibility']['seed']
    )
    
    # Criar DataFrame
    feature_names = [f'feature_{i}' for i in range(30)]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')
    
    logger.info(f"✅ Dados criados: {X.shape[0]} amostras, {X.shape[1]} features")
    logger.info(f"✅ Distribuição de classes: {y.value_counts().to_dict()}")
    
    return X, y

def train_all_models(X_train, y_train, X_test, y_test, config: dict):
    """Treina todos os modelos."""
    logger.info("\n" + "="*70)
    logger.info("TREINANDO TODOS OS MODELOS")
    logger.info("="*70)
    
    results = {}
    
    # 1. Regressão Logística (White Box)
    logger.info("\n### 1/5 - Regressão Logística (White Box) ###")
    try:
        from src.ml_models.train_logistic_regression import LogisticRegressionTrainer
        lr = LogisticRegressionTrainer()
        lr.train(X_train, y_train)
        lr_metrics = lr.evaluate(X_test, y_test)
        results['Regressão Logística'] = lr_metrics
        logger.info(f"✅ F1-Score: {lr_metrics['f1_score']:.4f} | AUC: {lr_metrics['auc']:.4f}")
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        results['Regressão Logística'] = None
    
    # 2. Árvore de Decisão (White Box)
    logger.info("\n### 2/5 - Árvore de Decisão (White Box) ###")
    try:
        from src.ml_models.train_decision_tree import DecisionTreeTrainer
        dt = DecisionTreeTrainer()
        dt.train(X_train, y_train, max_depth=config['ml_models']['decision_tree']['max_depth'])
        dt_metrics = dt.evaluate(X_test, y_test)
        results['Árvore de Decisão'] = dt_metrics
        logger.info(f"✅ F1-Score: {dt_metrics['f1_score']:.4f} | AUC: {dt_metrics['auc']:.4f}")
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        results['Árvore de Decisão'] = None
    
    # 3. XGBoost (Black Box)
    logger.info("\n### 3/5 - XGBoost (Black Box) ###")
    try:
        from src.ml_models.train_xgboost import XGBoostTrainer
        xgb = XGBoostTrainer(config)
        xgb.train(X_train, y_train)
        xgb_metrics = xgb.evaluate(X_test, y_test)
        results['XGBoost'] = xgb_metrics
        logger.info(f"✅ F1-Score: {xgb_metrics['f1_score']:.4f} | AUC: {xgb_metrics['auc']:.4f}")
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        results['XGBoost'] = None
    
    # 4. LightGBM (Black Box)
    logger.info("\n### 4/5 - LightGBM (Black Box) ###")
    try:
        from src.ml_models.train_lightgbm import LightGBMTrainer
        lgb = LightGBMTrainer(config)
        lgb.train(X_train, y_train)
        lgb_metrics = lgb.evaluate(X_test, y_test)
        results['LightGBM'] = lgb_metrics
        logger.info(f"✅ F1-Score: {lgb_metrics['f1_score']:.4f} | AUC: {lgb_metrics['auc']:.4f}")
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        results['LightGBM'] = None
    
    # 5. CatBoost (Black Box)
    logger.info("\n### 5/5 - CatBoost (Black Box) ###")
    try:
        from src.ml_models.train_catboost import CatBoostTrainer
        cat = CatBoostTrainer(config)
        cat.train(X_train, y_train)
        cat_metrics = cat.evaluate(X_test, y_test)
        results['CatBoost'] = cat_metrics
        logger.info(f"✅ F1-Score: {cat_metrics['f1_score']:.4f} | AUC: {cat_metrics['auc']:.4f}")
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        results['CatBoost'] = None
    
    return results

def run_comparison(X_train, y_train, X_test, y_test, config: dict):
    """Executa comparação completa."""
    logger.info("\n" + "="*70)
    logger.info("COMPARAÇÃO WHITE BOX vs BLACK BOX")
    logger.info("="*70)
    
    try:
        from src.ml_models.compare_white_black_box import WhiteBoxBlackBoxComparison
        
        comparison = WhiteBoxBlackBoxComparison()
        comparison.train_all_models(X_train, y_train, X_test, y_test)
        
        # Tabela comparativa
        df = comparison.get_comparison_table()
        logger.info("\n### Tabela Comparativa ###\n")
        logger.info(df.to_string(index=False))
        
        # Criar diretório de resultados
        results_dir = Path('results/comparison')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Gerar visualizações
        logger.info("\n### Gerando Visualizações ###")
        comparison.plot_comparison(save_path=str(results_dir / 'all_metrics.png'))
        logger.info(f"✅ Salvo: {results_dir / 'all_metrics.png'}")
        
        comparison.plot_f1_comparison(save_path=str(results_dir / 'f1_comparison.png'))
        logger.info(f"✅ Salvo: {results_dir / 'f1_comparison.png'}")
        
        # Gerar relatório
        logger.info("\n### Gerando Relatório ###")
        comparison.generate_report(save_path=str(results_dir / 'report.md'))
        logger.info(f"✅ Salvo: {results_dir / 'report.md'}")
        
        # Salvar resultados
        comparison.save_results(str(results_dir / 'results.json'))
        logger.info(f"✅ Salvo: {results_dir / 'results.json'}")
        
        logger.info(f"\n✅ Todos os resultados salvos em: {results_dir}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"❌ Erro na comparação: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Função principal."""
    logger.info("="*70)
    logger.info("FRAMEWORK MULTI-PARADIGMA - COMPARAÇÃO DE MODELOS ML")
    logger.info("="*70)
    
    try:
        # 1. Carregar configuração
        logger.info("\n### Carregando Configuração ###")
        config = load_config()
        logger.info("✅ Configuração carregada")
        
        # 2. Criar dados
        X, y = create_synthetic_data(config)
        
        # 3. Split
        logger.info("\n### Dividindo Dados ###")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['data_split']['test_size'],
            random_state=config['reproducibility']['seed'],
            stratify=y
        )
        logger.info(f"✅ Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
        
        # 4. Treinar todos os modelos
        results = train_all_models(X_train, y_train, X_test, y_test, config)
        
        # 5. Executar comparação
        comparison = run_comparison(X_train, y_train, X_test, y_test, config)
        
        # 6. Resumo final
        logger.info("\n" + "="*70)
        logger.info("RESUMO FINAL")
        logger.info("="*70)
        
        successful = sum(1 for v in results.values() if v is not None)
        total = len(results)
        
        logger.info(f"\n✅ Modelos treinados com sucesso: {successful}/{total}")
        
        if successful > 0:
            logger.info("\n### Melhores Resultados ###")
            valid_results = {k: v for k, v in results.items() if v is not None}
            best_f1 = max(valid_results.items(), key=lambda x: x[1]['f1_score'])
            best_auc = max(valid_results.items(), key=lambda x: x[1]['auc'])
            
            logger.info(f"🏆 Melhor F1-Score: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
            logger.info(f"🏆 Melhor AUC: {best_auc[0]} ({best_auc[1]['auc']:.4f})")
        
        logger.info("\n" + "="*70)
        logger.info("✅ EXECUÇÃO COMPLETA COM SUCESSO!")
        logger.info("="*70)
        logger.info("\n📂 Resultados disponíveis em: results/comparison/")
        logger.info("   - all_metrics.png")
        logger.info("   - f1_comparison.png")
        logger.info("   - report.md")
        logger.info("   - results.json")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ ERRO NA EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
