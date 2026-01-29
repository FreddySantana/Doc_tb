"""
Pipeline completo de Machine Learning

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-04-20
Última Modificação: 2025-07-25

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
Pipeline de Machine Learning

Orquestra o treinamento de múltiplos modelos (XGBoost, LightGBM, CatBoost)
e seleciona o melhor baseado em métricas de desempenho.

Conforme descrito na Seção 4.3 da tese.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
import json

from src.utils import setup_logger, load_config, load_data
from src.ml_models import XGBoostTrainer, LightGBMTrainer, CatBoostTrainer

logger = setup_logger(__name__)


class MLPipeline:
    """
    Pipeline de Machine Learning.
    
    Treina múltiplos modelos de boosting e seleciona o melhor
    baseado em F1-Score (métrica principal para dados desbalanceados).
    
    Attributes:
        config: Dicionário de configurações
        trainers: Dicionário com os treinadores
        best_model_name: Nome do melhor modelo
        best_model: Melhor modelo treinado
        results: Resultados de todos os modelos
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o pipeline.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        
        # Inicializar treinadores
        self.trainers = {
            'XGBoost': XGBoostTrainer(config),
            'LightGBM': LightGBMTrainer(config),
            'CatBoost': CatBoostTrainer(config)
        }
        
        self.best_model_name = None
        self.best_model = None
        self.results = {}
        
        self.output_dir = Path('results/ml_models')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info('MLPipeline inicializado')
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        use_cross_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Treina todos os modelos.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            use_cross_validation: Se deve usar validação cruzada
        
        Returns:
            Dicionário com todos os modelos treinados
        """
        logger.info('='*80)
        logger.info('TREINANDO TODOS OS MODELOS DE MACHINE LEARNING')
        logger.info('='*80)
        
        trained_models = {}
        
        for model_name, trainer in self.trainers.items():
            try:
                logger.info(f'\n{"="*80}')
                logger.info(f'Treinando {model_name}...')
                logger.info(f'{"="*80}')
                
                model = trainer.train(
                    X_train, y_train,
                    X_val, y_val,
                    use_cross_validation=use_cross_validation
                )
                
                trained_models[model_name] = {
                    'trainer': trainer,
                    'model': model
                }
                
                logger.info(f'✅ {model_name} treinado com sucesso!')
                
            except Exception as e:
                logger.error(f'❌ Erro ao treinar {model_name}: {e}')
                continue
        
        logger.info('\n' + '='*80)
        logger.info(f'✅ {len(trained_models)} modelos treinados com sucesso!')
        logger.info('='*80)
        
        return trained_models
    
    def evaluate_all_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Avalia todos os modelos treinados.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            threshold: Limiar de decisão
        
        Returns:
            Dicionário com métricas de todos os modelos
        """
        logger.info('='*80)
        logger.info('AVALIANDO TODOS OS MODELOS')
        logger.info('='*80)
        
        all_metrics = {}
        
        for model_name, trainer in self.trainers.items():
            if trainer.model is None:
                logger.warning(f'{model_name} não foi treinado. Pulando avaliação.')
                continue
            
            try:
                logger.info(f'\nAvaliando {model_name}...')
                metrics = trainer.evaluate(X_test, y_test, threshold)
                all_metrics[model_name] = metrics
                
            except Exception as e:
                logger.error(f'Erro ao avaliar {model_name}: {e}')
                continue
        
        self.results = all_metrics
        
        return all_metrics
    
    def select_best_model(
        self,
        metric: str = 'f1_score'
    ) -> Tuple[str, Any]:
        """
        Seleciona o melhor modelo baseado em uma métrica.
        
        Args:
            metric: Métrica para seleção (padrão: f1_score)
        
        Returns:
            Tupla (nome_do_modelo, modelo)
        """
        if not self.results:
            raise ValueError('Nenhum modelo foi avaliado ainda')
        
        logger.info('='*80)
        logger.info(f'SELECIONANDO MELHOR MODELO (Métrica: {metric})')
        logger.info('='*80)
        
        # Encontrar melhor modelo
        best_score = -1
        best_name = None
        
        for model_name, metrics in self.results.items():
            score = metrics.get(metric, 0)
            logger.info(f'{model_name}: {metric} = {score:.4f}')
            
            if score > best_score:
                best_score = score
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.trainers[best_name].model
        
        logger.info('\n' + '='*80)
        logger.info(f'🏆 MELHOR MODELO: {best_name}')
        logger.info(f'   {metric.upper()}: {best_score:.4f}')
        logger.info('='*80)
        
        return best_name, self.best_model
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Gera relatório comparativo de todos os modelos.
        
        Returns:
            DataFrame com comparação de métricas
        """
        if not self.results:
            raise ValueError('Nenhum modelo foi avaliado ainda')
        
        # Criar DataFrame comparativo
        comparison_df = pd.DataFrame(self.results).T
        
        # Ordenar por F1-Score
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        # Salvar relatório
        report_path = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(report_path)
        logger.info(f'Relatório comparativo salvo em {report_path}')
        
        return comparison_df
    
    def save_best_model(self) -> None:
        """Salva o melhor modelo."""
        if self.best_model_name is None:
            raise ValueError('Nenhum modelo foi selecionado ainda')
        
        # Salvar modelo
        trainer = self.trainers[self.best_model_name]
        trainer.save(f'best_model_{self.best_model_name.lower()}.pkl')
        
        # Salvar metadados
        metadata = {
            'best_model': self.best_model_name,
            'metrics': self.results[self.best_model_name],
            'all_results': self.results
        }
        
        metadata_path = self.output_dir / 'best_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f'Metadados do melhor modelo salvos em {metadata_path}')
    
    def run(
        self,
        input_train_path: str = 'data/processed/train_balanced.csv',
        input_test_path: str = 'data/processed/test.csv'
    ) -> Tuple[str, Any, Dict[str, Dict[str, float]]]:
        """
        Executa o pipeline completo de ML.
        
        Args:
            input_train_path: Caminho para dados de treino
            input_test_path: Caminho para dados de teste
        
        Returns:
            Tupla (nome_melhor_modelo, melhor_modelo, todas_métricas)
        """
        logger.info('='*80)
        logger.info('INICIANDO PIPELINE COMPLETO DE MACHINE LEARNING')
        logger.info('='*80)
        
        # Carregar dados
        logger.info(f'Carregando dados de treino: {input_train_path}')
        train_df = load_data(input_train_path)
        
        logger.info(f'Carregando dados de teste: {input_test_path}')
        test_df = load_data(input_test_path)
        
        target_col = self.config['target']['column_name']
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]
        
        logger.info(f'Dimensões - Treino: {X_train.shape}, Teste: {X_test.shape}')
        
        # Treinar todos os modelos
        self.train_all_models(X_train, y_train, use_cross_validation=True)
        
        # Avaliar todos os modelos
        all_metrics = self.evaluate_all_models(X_test, y_test)
        
        # Selecionar melhor modelo
        best_name, best_model = self.select_best_model(metric='f1_score')
        
        # Gerar relatório comparativo
        comparison_df = self.generate_comparison_report()
        
        logger.info('\n' + '='*80)
        logger.info('RELATÓRIO COMPARATIVO DE MODELOS')
        logger.info('='*80)
        logger.info('\n' + comparison_df.to_string())
        
        # Salvar melhor modelo
        self.save_best_model()
        
        logger.info('\n' + '='*80)
        logger.info('✅ PIPELINE DE MACHINE LEARNING CONCLUÍDO COM SUCESSO!')
        logger.info('='*80)
        
        return best_name, best_model, all_metrics


def main():
    """Função principal para execução standalone"""
    logger.info('Iniciando Pipeline de Machine Learning')
    
    # Carregar configuração
    config = load_config()
    
    # Criar e executar pipeline
    pipeline = MLPipeline(config)
    best_name, best_model, all_metrics = pipeline.run()
    
    print('\n' + '='*80)
    print('🏆 MELHOR MODELO SELECIONADO')
    print('='*80)
    print(f'\nModelo: {best_name}')
    print(f'\nMétricas:')
    for metric, value in all_metrics[best_name].items():
        if isinstance(value, float):
            print(f'  {metric}: {value:.4f}')
        else:
            print(f'  {metric}: {value}')
    
    print('\n' + '='*80)
    print('📁 RESULTADOS SALVOS')
    print('='*80)
    print(f'  - Modelos: results/ml_models/')
    print(f'  - Comparação: results/ml_models/model_comparison.csv')
    print(f'  - Melhor modelo: results/ml_models/best_model_{best_name.lower()}.pkl')
    print('='*80)


if __name__ == '__main__':
    main()
