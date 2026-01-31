"""
Módulo para explicabilidade usando LIME

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-06-08
Última Modificação: 2025-09-15

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
LIME Explainer

Implementa explicabilidade usando LIME (Local Interpretable Model-agnostic Explanations)
conforme descrito na Seção 4.4 da tese.

LIME cria aproximações locais interpretáveis de modelos complexos.

Referência: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import matplotlib.pyplot as plt

from lime import lime_tabular
from sklearn.base import BaseEstimator

from src.utils import setup_logger, load_config

logger = setup_logger(__name__)


class LIMEExplainer:
    """
    Explicador LIME para modelos de Machine Learning.
    
    LIME (Local Interpretable Model-agnostic Explanations) cria
    aproximações lineares locais de modelos complexos.
    
    Funcionamento:
    1. Gera perturbações da amostra a ser explicada
    2. Obtém predições do modelo para as perturbações
    3. Treina modelo linear local ponderado por proximidade
    4. Usa coeficientes do modelo linear como explicação
    
    Attributes:
        config: Dicionário de configurações
        model: Modelo a ser explicado
        explainer: Explainer LIME
        feature_names: Nomes das features
        class_names: Nomes das classes
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model: BaseEstimator,
        feature_names: List[str],
        class_names: List[str] = ['Não-Abandono', 'Abandono']
    ):
        """
        Inicializa o explainer.
        
        Args:
            config: Dicionário de configurações
            model: Modelo treinado a ser explicado
            feature_names: Lista com nomes das features
            class_names: Lista com nomes das classes
        """
        self.config = config
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        
        self.output_dir = Path('results/xai/lime')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info('LIMEExplainer inicializado')
    
    def fit(
        self,
        X_train: pd.DataFrame,
        mode: str = 'classification'
    ) -> None:
        """
        Cria o explainer LIME.
        
        Args:
            X_train: Features de treino (para estatísticas)
            mode: Modo de operação ('classification' ou 'regression')
        """
        logger.info('='*80)
        logger.info('CRIANDO EXPLAINER LIME')
        logger.info('='*80)
        
        # Criar explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            random_state=self.config.get('random_state', 42)
        )
        
        logger.info('✅ LIME Explainer criado com sucesso')
    
    def explain_instance(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Any:
        """
        Explica uma instância específica.
        
        Args:
            X: Features
            sample_idx: Índice da amostra a explicar
            num_features: Número de features na explicação
            num_samples: Número de amostras para aproximação local
        
        Returns:
            Objeto Explanation do LIME
        """
        if self.explainer is None:
            raise ValueError('Explainer não foi criado. Execute fit() primeiro.')
        
        logger.info('='*80)
        logger.info(f'EXPLICANDO INSTÂNCIA {sample_idx}')
        logger.info('='*80)
        
        # Obter amostra
        instance = X.iloc[sample_idx].values
        
        # Explicar
        logger.info(f'Gerando explicação com {num_samples} amostras...')
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Predição do modelo
        prediction_proba = self.model.predict_proba(X.iloc[[sample_idx]])[0, 1]
        prediction_class = int(prediction_proba >= 0.5)
        
        logger.info(f'✅ Explicação gerada')
        logger.info(f'   Probabilidade de abandono: {prediction_proba:.4f}')
        logger.info(f'   Classe predita: {self.class_names[prediction_class]}')
        
        return explanation
    
    def plot_explanation(
        self,
        explanation: Any,
        sample_idx: int = 0,
        save: bool = True
    ) -> None:
        """
        Gera gráfico da explicação.
        
        Args:
            explanation: Objeto Explanation do LIME
            sample_idx: Índice da amostra
            save: Se deve salvar o gráfico
        """
        logger.info(f'Gerando gráfico de explicação para amostra {sample_idx}...')
        
        # Gerar figura
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'lime_explanation_sample_{sample_idx}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Gráfico salvo em {save_path}')
        
        plt.close()
    
    def get_explanation_dict(
        self,
        explanation: Any,
        label: int = 1
    ) -> Dict[str, Any]:
        """
        Converte explicação para dicionário.
        
        Args:
            explanation: Objeto Explanation do LIME
            label: Classe a explicar (1 = Abandono)
        
        Returns:
            Dicionário com explicação
        """
        # Obter lista de (feature, weight)
        exp_list = explanation.as_list(label=label)
        
        # Separar contribuições positivas e negativas
        positive_contrib = [(f, w) for f, w in exp_list if w > 0]
        negative_contrib = [(f, w) for f, w in exp_list if w < 0]
        
        # Ordenar por magnitude
        positive_contrib = sorted(positive_contrib, key=lambda x: abs(x[1]), reverse=True)
        negative_contrib = sorted(negative_contrib, key=lambda x: abs(x[1]), reverse=True)
        
        explanation_dict = {
            'intercept': float(explanation.intercept[label]),
            'prediction_probability': float(explanation.predict_proba[label]),
            'local_prediction': float(explanation.local_pred[label]),
            'positive_contributions': [{'feature': f, 'weight': float(w)} for f, w in positive_contrib],
            'negative_contributions': [{'feature': f, 'weight': float(w)} for f, w in negative_contrib]
        }
        
        logger.info(f'\nExplicação LIME:')
        logger.info(f'  Intercept: {explanation_dict["intercept"]:.4f}')
        logger.info(f'  Probabilidade predita: {explanation_dict["prediction_probability"]:.4f}')
        logger.info(f'  Predição local: {explanation_dict["local_prediction"]:.4f}')
        
        logger.info(f'\nTop 5 contribuições positivas:')
        for contrib in positive_contrib[:5]:
            logger.info(f'  {contrib[0]}: {contrib[1]:.4f}')
        
        logger.info(f'\nTop 5 contribuições negativas:')
        for contrib in negative_contrib[:5]:
            logger.info(f'  {contrib[0]}: {contrib[1]:.4f}')
        
        return explanation_dict
    
    def explain_multiple_instances(
        self,
        X: pd.DataFrame,
        sample_indices: List[int],
        num_features: int = 10,
        num_samples: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Explica múltiplas instâncias.
        
        Args:
            X: Features
            sample_indices: Lista de índices a explicar
            num_features: Número de features na explicação
            num_samples: Número de amostras para aproximação local
        
        Returns:
            Lista de dicionários com explicações
        """
        logger.info('='*80)
        logger.info(f'EXPLICANDO {len(sample_indices)} INSTÂNCIAS')
        logger.info('='*80)
        
        explanations = []
        
        for idx in sample_indices:
            logger.info(f'\nExplicando amostra {idx}...')
            
            # Gerar explicação
            explanation = self.explain_instance(
                X, idx,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Converter para dicionário
            exp_dict = self.get_explanation_dict(explanation, label=1)
            exp_dict['sample_index'] = idx
            
            # Salvar gráfico
            self.plot_explanation(explanation, sample_idx=idx)
            
            explanations.append(exp_dict)
        
        logger.info(f'\n✅ {len(explanations)} instâncias explicadas')
        
        return explanations
    
    def compare_with_shap(
        self,
        X: pd.DataFrame,
        shap_importance: pd.DataFrame,
        num_features: int = 20
    ) -> pd.DataFrame:
        """
        Compara importância de features LIME vs SHAP.
        
        Args:
            X: Features
            shap_importance: DataFrame com importância SHAP
            num_features: Número de features a comparar
        
        Returns:
            DataFrame comparativo
        """
        logger.info('='*80)
        logger.info('COMPARANDO LIME vs SHAP')
        logger.info('='*80)
        
        # Calcular importância média LIME em várias amostras
        sample_indices = np.random.choice(len(X), size=min(50, len(X)), replace=False)
        
        lime_importance = {feat: 0.0 for feat in self.feature_names}
        
        for idx in sample_indices:
            explanation = self.explain_instance(X, idx, num_features=len(self.feature_names))
            exp_list = explanation.as_list(label=1)
            
            for feat, weight in exp_list:
                # Extrair nome da feature (LIME retorna com condição)
                feat_name = feat.split('<=')[0].split('>')[0].strip()
                if feat_name in lime_importance:
                    lime_importance[feat_name] += abs(weight)
        
        # Normalizar
        for feat in lime_importance:
            lime_importance[feat] /= len(sample_indices)
        
        # Criar DataFrame
        lime_df = pd.DataFrame({
            'feature': list(lime_importance.keys()),
            'lime_importance': list(lime_importance.values())
        }).sort_values('lime_importance', ascending=False)
        
        # Merge com SHAP
        comparison_df = lime_df.merge(
            shap_importance[['feature', 'importance']].rename(columns={'importance': 'shap_importance'}),
            on='feature',
            how='outer'
        ).fillna(0)
        
        # Calcular correlação
        correlation = comparison_df['lime_importance'].corr(comparison_df['shap_importance'])
        
        logger.info(f'\nCorrelação LIME vs SHAP: {correlation:.4f}')
        
        # Salvar
        save_path = self.output_dir / 'lime_vs_shap_comparison.csv'
        comparison_df.to_csv(save_path, index=False)
        logger.info(f'Comparação salva em {save_path}')
        
        return comparison_df.head(num_features)


def main():
    """Função principal para execução standalone"""
    from src.utils import load_data, load_model
    
    logger.info('Iniciando LIME Explainer')
    
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
    explainer = LIMEExplainer(
        config, model,
        feature_names=X_train.columns.tolist()
    )
    explainer.fit(X_train)
    
    # Explicar algumas instâncias
    sample_indices = [0, 1, 2, 3, 4]
    explanations = explainer.explain_multiple_instances(
        X_test,
        sample_indices,
        num_features=10
    )
    
    print('\n' + '='*80)
    print('✅ LIME EXPLAINER CONCLUÍDO COM SUCESSO!')
    print('='*80)
    print(f'\n📁 Resultados salvos em: {explainer.output_dir}')
    print(f'\n📊 {len(explanations)} instâncias explicadas')
    print('='*80)


if __name__ == '__main__':
    main()
