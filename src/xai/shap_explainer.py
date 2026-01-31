"""
Módulo para explicabilidade usando SHAP

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-06-05
Última Modificação: 2025-09-12

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
SHAP Explainer

Implementa explicabilidade usando SHAP (SHapley Additive exPlanations)
conforme descrito na Seção 4.4 da tese.

SHAP usa valores de Shapley da teoria dos jogos para atribuir
importância a cada feature na predição.

Referência: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.base import BaseEstimator

from src.utils import setup_logger, load_config

logger = setup_logger(__name__)


class SHAPExplainer:
    """
    Explicador SHAP para modelos de Machine Learning.
    
    SHAP (SHapley Additive exPlanations) calcula a contribuição
    de cada feature para a predição usando valores de Shapley.
    
    Propriedades dos valores SHAP:
    - Consistência local: features importantes têm valores SHAP maiores
    - Missingness: features ausentes têm valor SHAP zero
    - Consistência: mudanças no modelo refletem em mudanças nos valores SHAP
    
    Attributes:
        config: Dicionário de configurações
        model: Modelo a ser explicado
        explainer: Explainer SHAP
        shap_values: Valores SHAP calculados
    """
    
    def __init__(self, config: Dict[str, Any], model: BaseEstimator):
        """
        Inicializa o explainer.
        
        Args:
            config: Dicionário de configurações
            model: Modelo treinado a ser explicado
        """
        self.config = config
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
        self.output_dir = Path('results/xai/shap')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info('SHAPExplainer inicializado')
    
    def fit(self, X_train: pd.DataFrame, max_samples: int = 100) -> None:
        """
        Cria o explainer SHAP.
        
        Args:
            X_train: Features de treino (para background data)
            max_samples: Número máximo de amostras para background
        """
        logger.info('='*80)
        logger.info('CRIANDO EXPLAINER SHAP')
        logger.info('='*80)
        
        # Usar TreeExplainer para modelos baseados em árvores
        try:
            logger.info('Tentando TreeExplainer (para modelos baseados em árvores)...')
            self.explainer = shap.TreeExplainer(self.model)
            logger.info('✅ TreeExplainer criado com sucesso')
        except Exception as e:
            logger.warning(f'TreeExplainer falhou: {e}')
            logger.info('Usando KernelExplainer (mais lento, mas funciona para qualquer modelo)...')
            
            # Usar amostra menor para KernelExplainer (mais lento)
            background = shap.sample(X_train, min(max_samples, len(X_train)))
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
            logger.info('✅ KernelExplainer criado com sucesso')
    
    def explain(
        self,
        X: pd.DataFrame,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Calcula valores SHAP para as amostras.
        
        Args:
            X: Features a serem explicadas
            max_samples: Número máximo de amostras a explicar
        
        Returns:
            Array com valores SHAP
        """
        if self.explainer is None:
            raise ValueError('Explainer não foi criado. Execute fit() primeiro.')
        
        logger.info('='*80)
        logger.info('CALCULANDO VALORES SHAP')
        logger.info('='*80)
        
        # Limitar número de amostras se necessário
        if max_samples is not None and len(X) > max_samples:
            logger.info(f'Limitando a {max_samples} amostras para explicação')
            X = X.sample(n=max_samples, random_state=self.config.get('random_state', 42))
        
        logger.info(f'Calculando valores SHAP para {len(X)} amostras...')
        
        # Calcular valores SHAP
        self.shap_values = self.explainer.shap_values(X)
        
        # Para classificação binária, pegar valores da classe positiva
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        # Salvar expected value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            self.expected_value = self.explainer.expected_value[1]
        else:
            self.expected_value = self.explainer.expected_value
        
        logger.info(f'✅ Valores SHAP calculados: {self.shap_values.shape}')
        logger.info(f'   Expected value: {self.expected_value:.4f}')
        
        return self.shap_values
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        save: bool = True
    ) -> None:
        """
        Gera gráfico de resumo SHAP.
        
        Args:
            X: Features originais
            max_display: Número máximo de features a exibir
            save: Se deve salvar o gráfico
        """
        if self.shap_values is None:
            raise ValueError('Valores SHAP não foram calculados. Execute explain() primeiro.')
        
        logger.info('Gerando gráfico de resumo SHAP...')
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'shap_summary_plot.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def plot_waterfall(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        save: bool = True
    ) -> None:
        """
        Gera gráfico waterfall para uma amostra específica.
        
        Args:
            X: Features originais
            sample_idx: Índice da amostra
            save: Se deve salvar o gráfico
        """
        if self.shap_values is None:
            raise ValueError('Valores SHAP não foram calculados. Execute explain() primeiro.')
        
        logger.info(f'Gerando gráfico waterfall para amostra {sample_idx}...')
        
        # Criar Explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            data=X.iloc[sample_idx].values,
            feature_names=X.columns.tolist()
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'shap_waterfall_sample_{sample_idx}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def plot_force(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        save: bool = True
    ) -> None:
        """
        Gera gráfico de força para uma amostra específica.
        
        Args:
            X: Features originais
            sample_idx: Índice da amostra
            save: Se deve salvar o gráfico
        """
        if self.shap_values is None:
            raise ValueError('Valores SHAP não foram calculados. Execute explain() primeiro.')
        
        logger.info(f'Gerando gráfico de força para amostra {sample_idx}...')
        
        # Gerar plot de força
        shap.force_plot(
            self.expected_value,
            self.shap_values[sample_idx],
            X.iloc[sample_idx],
            matplotlib=True,
            show=False
        )
        
        if save:
            save_path = self.output_dir / f'shap_force_sample_{sample_idx}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def get_feature_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Calcula importância global das features baseada em valores SHAP.
        
        Args:
            X: Features originais
            top_n: Número de features mais importantes
        
        Returns:
            DataFrame com features e importâncias
        """
        if self.shap_values is None:
            raise ValueError('Valores SHAP não foram calculados. Execute explain() primeiro.')
        
        # Calcular importância média absoluta
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Criar DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f'\nTop {top_n} features mais importantes (SHAP):')
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f'  {row["feature"]}: {row["importance"]:.4f}')
        
        # Salvar
        save_path = self.output_dir / 'shap_feature_importance.csv'
        importance_df.to_csv(save_path, index=False)
        logger.info(f'Importância salva em {save_path}')
        
        return importance_df.head(top_n)
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Explica uma predição específica.
        
        Args:
            X: Features originais
            sample_idx: Índice da amostra
        
        Returns:
            Dicionário com explicação detalhada
        """
        if self.shap_values is None:
            raise ValueError('Valores SHAP não foram calculados. Execute explain() primeiro.')
        
        # Predição do modelo
        prediction_proba = self.model.predict_proba(X.iloc[[sample_idx]])[0, 1]
        prediction_class = int(prediction_proba >= 0.5)
        
        # Valores SHAP da amostra
        sample_shap = self.shap_values[sample_idx]
        
        # Criar DataFrame com contribuições
        contributions = pd.DataFrame({
            'feature': X.columns,
            'value': X.iloc[sample_idx].values,
            'shap_value': sample_shap
        }).sort_values('shap_value', key=abs, ascending=False)
        
        explanation = {
            'sample_index': sample_idx,
            'prediction_probability': float(prediction_proba),
            'prediction_class': prediction_class,
            'expected_value': float(self.expected_value),
            'top_positive_contributions': contributions[contributions['shap_value'] > 0].head(5).to_dict('records'),
            'top_negative_contributions': contributions[contributions['shap_value'] < 0].head(5).to_dict('records')
        }
        
        logger.info(f'\nExplicação da predição para amostra {sample_idx}:')
        logger.info(f'  Probabilidade de abandono: {prediction_proba:.4f}')
        logger.info(f'  Classe predita: {"Abandono" if prediction_class == 1 else "Não-Abandono"}')
        logger.info(f'  Expected value: {self.expected_value:.4f}')
        
        return explanation


def main():
    """Função principal para execução standalone"""
    from src.utils import load_data, load_model
    
    logger.info('Iniciando SHAP Explainer')
    
    # Carregar configuração
    config = load_config()
    
    # Carregar modelo treinado
    logger.info('Carregando modelo...')
    model = load_model('results/ml_models/xgboost/xgboost_model.pkl')
    
    # Carregar dados
    logger.info('Carregando dados...')
    train_df = load_data('data/processed/train_balanced.csv')
    test_df = load_data('data/processed/test.csv')
    
    target_col = config['target']['column_name']
    X_train = train_df.drop(target_col, axis=1)
    X_test = test_df.drop(target_col, axis=1)
    
    # Criar explainer
    explainer = SHAPExplainer(config, model)
    explainer.fit(X_train, max_samples=100)
    
    # Explicar amostras de teste
    explainer.explain(X_test, max_samples=200)
    
    # Gerar visualizações
    explainer.plot_summary(X_test)
    explainer.plot_waterfall(X_test, sample_idx=0)
    explainer.plot_force(X_test, sample_idx=0)
    
    # Obter importância das features
    importance_df = explainer.get_feature_importance(X_test, top_n=20)
    
    # Explicar predição específica
    explanation = explainer.explain_prediction(X_test, sample_idx=0)
    
    print('\n' + '='*80)
    print('✅ SHAP EXPLAINER CONCLUÍDO COM SUCESSO!')
    print('='*80)
    print(f'\n📁 Resultados salvos em: {explainer.output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
