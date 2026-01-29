"""
Módulo para treinamento de modelo LightGBM

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-04-12
Última Modificação: 2025-07-18

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
Treinamento de Modelo LightGBM

Implementa treinamento de LightGBM conforme descrito na Seção 4.3 da tese.
LightGBM é otimizado para velocidade e eficiência em grandes datasets.

Referência: Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import json

import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.utils import setup_logger, save_model, load_config

logger = setup_logger(__name__)


class LightGBMTrainer:
    """
    Treinador de modelo LightGBM.
    
    LightGBM usa gradient-based one-side sampling (GOSS) e
    exclusive feature bundling (EFB) para eficiência.
    
    Principais características:
    - Crescimento de árvore leaf-wise (mais profundo)
    - GOSS: mantém instâncias com gradientes grandes
    - EFB: agrupa features mutuamente exclusivas
    
    Attributes:
        config: Dicionário de configurações
        model: Modelo LightGBM treinado
        feature_importance: Importância das features
    """
    
    def __init__(self, config: Dict[str, Any] = None, random_state: int = None):
        """
        Inicializa o treinador.
        
        Args:
            config: Dicionário de configurações (opcional)
            random_state: Seed para reprodutibilidade (opcional)
        """
        # Aceitar tanto dict quanto int para compatibilidade
        if isinstance(config, int):
            random_state = config
            config = {}
        elif config is None:
            config = {}
        
        self.config = config
        self.random_state = random_state if random_state is not None else config.get('random_state', 42)
        
        # Hiperparâmetros padrão otimizados
        self.params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
            'is_unbalance': True,  # Para dados desbalanceados
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 20
        }
        
        self.model = None
        self.feature_importance = None
        self.training_history = {}
        
        self.output_dir = Path('results/ml_models/lightgbm')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info('LightGBMTrainer inicializado')
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_cross_validation: bool = False,
        cv_folds: int = 5
    ) -> lgb.LGBMClassifier:
        """
        Treina o modelo LightGBM.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            use_cross_validation: Se deve usar validação cruzada
            cv_folds: Número de folds para CV
        
        Returns:
            Modelo treinado
        """
        logger.info('='*80)
        logger.info('TREINANDO MODELO LIGHTGBM')
        logger.info('='*80)
        
        logger.info(f'Dimensões do conjunto de treino: {X_train.shape}')
        logger.info(f'Distribuição da classe alvo:')
        logger.info(f'  Classe 0: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)')
        logger.info(f'  Classe 1: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)')
        
        # Criar modelo
        self.model = lgb.LGBMClassifier(**self.params)
        
        # Validação cruzada (opcional)
        if use_cross_validation:
            logger.info(f'Executando validação cruzada com {cv_folds} folds...')
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=cv, scoring='f1', n_jobs=-1
            )
            logger.info(f'F1-Score CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
            self.training_history['cv_scores'] = cv_scores.tolist()
        
        # Treinar modelo
        if X_val is not None and y_val is not None:
            logger.info('Treinando com conjunto de validação...')
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss'
            )
            
            # Salvar histórico de treinamento
            if hasattr(self.model, 'evals_result_'):
                self.training_history['val_loss'] = self.model.evals_result_['valid_0']['binary_logloss']
        else:
            logger.info('Treinando sem conjunto de validação...')
            self.model.fit(X_train, y_train)
        
        # Extrair importância das features
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info('Modelo treinado com sucesso!')
        logger.info(f'Número de árvores: {self.model.n_estimators}')
        
        # Salvar top 10 features mais importantes
        logger.info('\nTop 10 features mais importantes:')
        for idx, row in self.feature_importance.head(10).iterrows():
            logger.info(f'  {row["feature"]}: {row["importance"]:.4f}')
        
        return self.model
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Avalia o modelo.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            threshold: Limiar de decisão
        
        Returns:
            Dicionário com métricas
        """
        logger.info('='*80)
        logger.info('AVALIANDO MODELO LIGHTGBM')
        logger.info('='*80)
        
        # Predições
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob)
        }
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        logger.info('Métricas de Desempenho:')
        logger.info(f'  Acurácia: {metrics["accuracy"]:.4f}')
        logger.info(f'  Precisão: {metrics["precision"]:.4f}')
        logger.info(f'  Recall (Sensibilidade): {metrics["recall"]:.4f}')
        logger.info(f'  F1-Score: {metrics["f1_score"]:.4f}')
        logger.info(f'  ROC-AUC: {metrics["auc"]:.4f}')
        logger.info(f'  Especificidade: {metrics["specificity"]:.4f}')
        
        logger.info('\nMatriz de Confusão:')
        logger.info(f'  TN: {tn}  FP: {fp}')
        logger.info(f'  FN: {fn}  TP: {tp}')
        
        # Relatório de classificação
        logger.info('\nRelatório de Classificação:')
        logger.info('\n' + classification_report(y_test, y_pred, target_names=['Não-Abandono', 'Abandono']))
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retorna as features mais importantes.
        
        Args:
            top_n: Número de features a retornar
        
        Returns:
            DataFrame com features e importâncias
        """
        if self.feature_importance is None:
            raise ValueError('Modelo não foi treinado ainda')
        
        return self.feature_importance.head(top_n)
    
    def save(self, filename: str = 'lightgbm_model.pkl') -> None:
        """
        Salva o modelo treinado.
        
        Args:
            filename: Nome do arquivo
        """
        if self.model is None:
            raise ValueError('Modelo não foi treinado ainda')
        
        # Salvar modelo
        model_path = self.output_dir / filename
        save_model(self.model, model_path)
        logger.info(f'Modelo salvo em {model_path}')
        
        # Salvar importância das features
        if self.feature_importance is not None:
            importance_path = self.output_dir / 'feature_importance.csv'
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f'Importância das features salva em {importance_path}')
        
        # Salvar histórico de treinamento
        if self.training_history:
            history_path = self.output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            logger.info(f'Histórico de treinamento salvo em {history_path}')
        
        # Salvar hiperparâmetros
        params_path = self.output_dir / 'hyperparameters.json'
        with open(params_path, 'w') as f:
            json.dump(self.params, f, indent=2)
        logger.info(f'Hiperparâmetros salvos em {params_path}')


def main():
    """Função principal para execução standalone"""
    from src.utils import load_data
    
    logger.info('Iniciando treinamento de LightGBM')
    
    # Carregar configuração
    config = load_config()
    trainer = LightGBMTrainer(config)
    
    # Carregar dados
    logger.info('Carregando dados...')
    train_df = load_data('data/processed/train_balanced.csv')
    test_df = load_data('data/processed/test.csv')
    
    target_col = config['target']['column_name']
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    
    # Treinar
    trainer.train(X_train, y_train, use_cross_validation=True)
    
    # Avaliar
    metrics = trainer.evaluate(X_test, y_test)
    
    # Salvar
    trainer.save()
    
    # Exibir top features
    print('\n' + '='*80)
    print('TOP 20 FEATURES MAIS IMPORTANTES')
    print('='*80)
    print(trainer.get_feature_importance(top_n=20).to_string(index=False))
    
    print('\n' + '='*80)
    print('✅ TREINAMENTO DE LIGHTGBM CONCLUÍDO COM SUCESSO!')
    print('='*80)
    print(f'\n📊 Métricas Finais:')
    print(f'   Acurácia: {metrics["accuracy"]:.4f}')
    print(f'   Precisão: {metrics["precision"]:.4f}')
    print(f'   Recall: {metrics["recall"]:.4f}')
    print(f'   F1-Score: {metrics["f1_score"]:.4f}')
    print(f'   ROC-AUC: {metrics["auc"]:.4f}')
    print(f'\n📁 Resultados salvos em: {trainer.output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
