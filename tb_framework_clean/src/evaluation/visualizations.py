"""
Módulo para geração de visualizações e gráficos

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-07-05
Última Modificação: 2025-09-22

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
Visualizações

Módulo para geração de visualizações e gráficos
conforme descrito na Seção 4.6 da tese.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from src.utils import setup_logger, load_config

logger = setup_logger(__name__)

# Configurar estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class Visualizer:
    """
    Gerador de visualizações.
    
    Cria gráficos profissionais para análise e apresentação:
    - Matriz de confusão
    - Curva ROC
    - Curva Precision-Recall
    - Importância de features
    - Comparação de modelos
    
    Attributes:
        config: Dicionário de configurações
        output_dir: Diretório de saída
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o visualizador.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        
        self.output_dir = Path('results/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info('Visualizer inicializado')
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'model',
        class_names: List[str] = ['Não-Abandono', 'Abandono'],
        save: bool = True
    ) -> None:
        """
        Plota matriz de confusão.
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            model_name: Nome do modelo
            class_names: Nomes das classes
            save: Se deve salvar o gráfico
        """
        logger.info(f'Gerando matriz de confusão para {model_name}...')
        
        # Calcular matriz
        cm = confusion_matrix(y_true, y_pred)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot com seaborn
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Contagem'},
            ax=ax
        )
        
        ax.set_xlabel('Predito', fontsize=14, fontweight='bold')
        ax.set_ylabel('Verdadeiro', fontsize=14, fontweight='bold')
        ax.set_title(f'Matriz de Confusão - {model_name}', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'model',
        save: bool = True
    ) -> None:
        """
        Plota curva ROC.
        
        Args:
            y_true: Valores verdadeiros
            y_proba: Probabilidades
            model_name: Nome do modelo
            save: Se deve salvar o gráfico
        """
        logger.info(f'Gerando curva ROC para {model_name}...')
        
        # Calcular curva ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot da curva
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
        ax.set_title(f'Receiver Operating Characteristic (ROC) - {model_name}', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = 'model',
        save: bool = True
    ) -> None:
        """
        Plota curva Precision-Recall.
        
        Args:
            y_true: Valores verdadeiros
            y_proba: Probabilidades
            model_name: Nome do modelo
            save: Se deve salvar o gráfico
        """
        logger.info(f'Gerando curva Precision-Recall para {model_name}...')
        
        # Calcular curva
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(y_true, y_proba)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot da curva
        ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'pr_curve_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str = 'model',
        top_n: int = 20,
        save: bool = True
    ) -> None:
        """
        Plota importância de features.
        
        Args:
            importance_df: DataFrame com features e importâncias
            model_name: Nome do modelo
            top_n: Número de features a exibir
            save: Se deve salvar o gráfico
        """
        logger.info(f'Gerando gráfico de importância de features para {model_name}...')
        
        # Pegar top N
        top_features = importance_df.head(top_n).copy()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot horizontal
        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        
        ax.set_xlabel('Importância', fontsize=14, fontweight='bold')
        ax.set_ylabel('Features', fontsize=14, fontweight='bold')
        ax.set_title(f'Top {top_n} Features Mais Importantes - {model_name}', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def plot_models_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'F1-Score',
        save: bool = True
    ) -> None:
        """
        Plota comparação de modelos.
        
        Args:
            comparison_df: DataFrame comparativo
            metric: Métrica principal para destacar
            save: Se deve salvar o gráfico
        """
        logger.info('Gerando gráfico de comparação de modelos...')
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
        
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Ordenar por métrica
            sorted_df = comparison_df.sort_values(metric_name, ascending=True)
            
            # Cores: destacar a métrica principal
            colors = ['steelblue' if metric_name != metric else 'darkorange' for _ in range(len(sorted_df))]
            
            # Plot
            ax.barh(range(len(sorted_df)), sorted_df[metric_name], color=colors)
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels(sorted_df['Model'])
            ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(metric_name, fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(True, axis='x', alpha=0.3)
            
            # Adicionar valores
            for i, v in enumerate(sorted_df[metric_name]):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.suptitle('Comparação de Modelos - Todas as Métricas', fontsize=18, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'models_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        model_name: str = 'model',
        save: bool = True
    ) -> None:
        """
        Plota histórico de treinamento.
        
        Args:
            history: Dicionário com histórico (ex: {'loss': [...], 'val_loss': [...]})
            model_name: Nome do modelo
            save: Se deve salvar o gráfico
        """
        logger.info(f'Gerando gráfico de histórico de treinamento para {model_name}...')
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot de cada métrica
        for metric_name, values in history.items():
            ax.plot(values, label=metric_name, linewidth=2)
        
        ax.set_xlabel('Iteração', fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=14, fontweight='bold')
        ax.set_title(f'Histórico de Treinamento - {model_name}', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'training_history_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def create_full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        importance_df: pd.DataFrame,
        model_name: str = 'model'
    ) -> None:
        """
        Cria relatório visual completo.
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            y_proba: Probabilidades
            importance_df: DataFrame com importância de features
            model_name: Nome do modelo
        """
        logger.info('='*80)
        logger.info(f'GERANDO RELATÓRIO VISUAL COMPLETO: {model_name.upper()}')
        logger.info('='*80)
        
        # Gerar todos os gráficos
        self.plot_confusion_matrix(y_true, y_pred, model_name)
        self.plot_roc_curve(y_true, y_proba, model_name)
        self.plot_precision_recall_curve(y_true, y_proba, model_name)
        self.plot_feature_importance(importance_df, model_name)
        
        logger.info(f'\n✅ Relatório visual completo gerado em {self.output_dir}')


def main():
    """Função principal para execução standalone"""
    from src.utils import load_data, load_model
    
    logger.info('Iniciando geração de visualizações')
    
    # Carregar configuração
    config = load_config()
    visualizer = Visualizer(config)
    
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
    
    # Importância de features
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Gerar relatório completo
    visualizer.create_full_report(
        y_test, y_pred, y_proba,
        importance_df,
        model_name='XGBoost'
    )
    
    print('\n' + '='*80)
    print('✅ VISUALIZAÇÕES GERADAS COM SUCESSO!')
    print('='*80)
    print(f'\n📁 Gráficos salvos em: {visualizer.output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
