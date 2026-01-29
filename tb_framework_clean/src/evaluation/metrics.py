"""
Módulo de métricas de avaliação (F1-Score, AUC, Precision, Recall)

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-07-01
Última Modificação: 2025-09-20

Descrição:
    Este módulo faz parte do framework multi-paradigma desenvolvido para predição
    de abandono de tratamento em pacientes com tuberculose. O framework integra
    técnicas de Machine Learning, Deep Reinforcement Learning, Natural Language
    Processing e Explainable AI.

Licença: MIT
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avaliação de Métricas

Módulo para cálculo e análise de métricas de desempenho
conforme descrito na Seção 4.6 da tese.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

from src.utils import setup_logger, load_config

logger = setup_logger(__name__)


class MetricsEvaluator:
    """
    Avaliador de métricas de desempenho.
    
    Calcula métricas padrão para classificação binária:
    - Acurácia, Precisão, Recall, F1-Score
    - ROC-AUC, Especificidade, Sensibilidade
    - Matriz de confusão
    - Curvas ROC e Precision-Recall
    
    Attributes:
        config: Dicionário de configurações
        results: Dicionário com todos os resultados
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o avaliador.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        self.results = {}
        
        self.output_dir = Path('results/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info('MetricsEvaluator inicializado')
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = 'model'
    ) -> Dict[str, Any]:
        """
        Calcula todas as métricas.
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições (classes)
            y_proba: Probabilidades (opcional)
            model_name: Nome do modelo
        
        Returns:
            Dicionário com todas as métricas
        """
        logger.info('='*80)
        logger.info(f'AVALIANDO MODELO: {model_name.upper()}')
        logger.info('='*80)
        
        metrics = {}
        
        # Métricas básicas
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        # Especificidade e sensibilidade
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = metrics['recall']  # Sensibilidade = Recall
        
        # Métricas baseadas em probabilidade (se disponível)
        if y_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            metrics['average_precision'] = float(average_precision_score(y_true, y_proba))
            
            # Curvas
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }
            
            metrics['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        
        # Log das métricas
        logger.info('\nMétricas de Desempenho:')
        logger.info(f'  Acurácia: {metrics["accuracy"]:.4f}')
        logger.info(f'  Precisão: {metrics["precision"]:.4f}')
        logger.info(f'  Recall (Sensibilidade): {metrics["recall"]:.4f}')
        logger.info(f'  F1-Score: {metrics["f1_score"]:.4f}')
        logger.info(f'  Especificidade: {metrics["specificity"]:.4f}')
        
        if 'roc_auc' in metrics:
            logger.info(f'  ROC-AUC: {metrics["roc_auc"]:.4f}')
            logger.info(f'  Average Precision: {metrics["average_precision"]:.4f}')
        
        logger.info('\nMatriz de Confusão:')
        logger.info(f'  TN: {tn}  FP: {fp}')
        logger.info(f'  FN: {fn}  TP: {tp}')
        
        # Salvar resultados
        self.results[model_name] = metrics
        
        return metrics
    
    def compare_models(
        self,
        models_metrics: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compara métricas de múltiplos modelos.
        
        Args:
            models_metrics: Dicionário {nome_modelo: métricas}
        
        Returns:
            DataFrame comparativo
        """
        logger.info('='*80)
        logger.info('COMPARANDO MODELOS')
        logger.info('='*80)
        
        # Extrair métricas principais
        comparison_data = []
        
        for model_name, metrics in models_metrics.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'Specificity': metrics.get('specificity', 0),
                'ROC-AUC': metrics.get('roc_auc', 0)
            }
            comparison_data.append(row)
        
        # Criar DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        # Log
        logger.info('\nComparação de Modelos:')
        logger.info('\n' + comparison_df.to_string(index=False))
        
        # Salvar
        save_path = self.output_dir / 'models_comparison.csv'
        comparison_df.to_csv(save_path, index=False)
        logger.info(f'\nComparação salva em {save_path}')
        
        return comparison_df
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'model',
        target_names: List[str] = ['Não-Abandono', 'Abandono']
    ) -> str:
        """
        Gera relatório de classificação detalhado.
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            model_name: Nome do modelo
            target_names: Nomes das classes
        
        Returns:
            String com relatório
        """
        logger.info('='*80)
        logger.info(f'RELATÓRIO DE CLASSIFICAÇÃO: {model_name.upper()}')
        logger.info('='*80)
        
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            digits=4
        )
        
        logger.info('\n' + report)
        
        # Salvar
        save_path = self.output_dir / f'{model_name}_classification_report.txt'
        with open(save_path, 'w') as f:
            f.write(f'Classification Report: {model_name}\n')
            f.write('='*80 + '\n\n')
            f.write(report)
        
        logger.info(f'Relatório salvo em {save_path}')
        
        return report
    
    def calculate_clinical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'model'
    ) -> Dict[str, float]:
        """
        Calcula métricas clínicas específicas para TB.
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            model_name: Nome do modelo
        
        Returns:
            Dicionário com métricas clínicas
        """
        logger.info('='*80)
        logger.info(f'MÉTRICAS CLÍNICAS: {model_name.upper()}')
        logger.info('='*80)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Métricas clínicas
        clinical_metrics = {}
        
        # Positive Predictive Value (PPV) = Precisão
        clinical_metrics['ppv'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        
        # Negative Predictive Value (NPV)
        clinical_metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        
        # Likelihood Ratio Positive (LR+)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        clinical_metrics['lr_positive'] = float(sensitivity / (1 - specificity)) if specificity < 1 else float('inf')
        
        # Likelihood Ratio Negative (LR-)
        clinical_metrics['lr_negative'] = float((1 - sensitivity) / specificity) if specificity > 0 else 0.0
        
        # Number Needed to Screen (NNS) - aproximação
        prevalence = (tp + fn) / (tn + fp + fn + tp)
        clinical_metrics['prevalence'] = float(prevalence)
        clinical_metrics['nns'] = float(1 / (sensitivity * prevalence)) if (sensitivity * prevalence) > 0 else float('inf')
        
        # Log
        logger.info('\nMétricas Clínicas:')
        logger.info(f'  PPV (Valor Preditivo Positivo): {clinical_metrics["ppv"]:.4f}')
        logger.info(f'  NPV (Valor Preditivo Negativo): {clinical_metrics["npv"]:.4f}')
        logger.info(f'  LR+ (Razão de Verossimilhança Positiva): {clinical_metrics["lr_positive"]:.4f}')
        logger.info(f'  LR- (Razão de Verossimilhança Negativa): {clinical_metrics["lr_negative"]:.4f}')
        logger.info(f'  Prevalência: {clinical_metrics["prevalence"]:.4f}')
        logger.info(f'  NNS (Número Necessário para Rastrear): {clinical_metrics["nns"]:.2f}')
        
        return clinical_metrics
    
    def save_all_results(self, filename: str = 'evaluation_results.json') -> None:
        """
        Salva todos os resultados.
        
        Args:
            filename: Nome do arquivo
        """
        save_path = self.output_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f'Todos os resultados salvos em {save_path}')


def main():
    """Função principal para execução standalone"""
    from src.utils import load_data, load_model
    
    logger.info('Iniciando avaliação de métricas')
    
    # Carregar configuração
    config = load_config()
    evaluator = MetricsEvaluator(config)
    
    # Carregar dados e modelo
    logger.info('Carregando dados...')
    test_df = load_data('data/processed/test.csv')
    
    target_col = config['target']['column_name']
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col].values
    
    logger.info('Carregando modelo...')
    model = load_model('results/ml_models/xgboost/xgboost_model.pkl')
    
    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Avaliar
    metrics = evaluator.evaluate(y_test, y_pred, y_proba, model_name='XGBoost')
    
    # Relatório de classificação
    evaluator.generate_classification_report(y_test, y_pred, model_name='XGBoost')
    
    # Métricas clínicas
    clinical_metrics = evaluator.calculate_clinical_metrics(y_test, y_pred, model_name='XGBoost')
    
    # Salvar
    evaluator.save_all_results()
    
    print('\n' + '='*80)
    print('✅ AVALIAÇÃO DE MÉTRICAS CONCLUÍDA COM SUCESSO!')
    print('='*80)
    print(f'\n📁 Resultados salvos em: {evaluator.output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
