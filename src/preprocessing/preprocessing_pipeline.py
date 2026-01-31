"""
Pipeline completo de pré-processamento de dados

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-03-01
Última Modificação: 2025-03-20

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
Pipeline Completo de Pré-processamento

Orquestra todas as etapas de pré-processamento em um pipeline unificado
conforme descrito na Seção 4.2 da tese.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

from src.utils import setup_logger, load_data, save_data
from src.preprocessing import (
    ExploratoryAnalysis,
    MissingValuesHandler,
    CategoricalEncoder,
    CorrelationHandler,
    OutliersHandler,
    ClassBalancer
)

logger = setup_logger(__name__)


class PreprocessingPipeline:
    """
    Pipeline completo de pré-processamento.
    
    Executa todas as etapas em sequência:
    1. Análise Exploratória
    2. Tratamento de Valores Ausentes (MICE)
    3. Codificação Categórica (Target Encoding)
    4. Tratamento de Correlação
    5. Detecção de Outliers (Isolation Forest)
    6. Balanceamento de Classes (SMOTE)
    
    Attributes:
        config: Dicionário de configurações
        steps: Lista de etapas do pipeline
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o pipeline.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        self.output_dir = Path('data/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.exploratory = ExploratoryAnalysis(config)
        self.missing_handler = MissingValuesHandler(config)
        self.categorical_encoder = CategoricalEncoder(config)
        self.correlation_handler = CorrelationHandler(config)
        self.outliers_handler = OutliersHandler(config)
        self.class_balancer = ClassBalancer(config)
        
        logger.info('PreprocessingPipeline inicializado')
    
    def run_exploratory_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executa análise exploratória.
        
        Args:
            df: DataFrame com dados brutos
        
        Returns:
            Dicionário com resultados da análise
        """
        logger.info('='*80)
        logger.info('ETAPA 1/6: ANÁLISE EXPLORATÓRIA')
        logger.info('='*80)
        
        results = self.exploratory.run()
        
        return results
    
    def run_missing_values_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa tratamento de valores ausentes.
        
        Args:
            df: DataFrame com dados brutos
        
        Returns:
            DataFrame com valores imputados
        """
        logger.info('='*80)
        logger.info('ETAPA 2/6: TRATAMENTO DE VALORES AUSENTES')
        logger.info('='*80)
        
        df_imputed = self.missing_handler.fit_transform(df)
        
        # Salvar checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_01_imputed.csv'
        save_data(df_imputed, checkpoint_path)
        logger.info(f'Checkpoint salvo: {checkpoint_path}')
        
        return df_imputed
    
    def run_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa codificação categórica.
        
        Args:
            df: DataFrame com valores imputados
        
        Returns:
            DataFrame com variáveis codificadas
        """
        logger.info('='*80)
        logger.info('ETAPA 3/6: CODIFICAÇÃO CATEGÓRICA')
        logger.info('='*80)
        
        df_encoded = self.categorical_encoder.fit_transform(df)
        
        # Salvar checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_02_encoded.csv'
        save_data(df_encoded, checkpoint_path)
        logger.info(f'Checkpoint salvo: {checkpoint_path}')
        
        return df_encoded
    
    def run_correlation_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa tratamento de correlação.
        
        Args:
            df: DataFrame com variáveis codificadas
        
        Returns:
            DataFrame sem variáveis correlacionadas
        """
        logger.info('='*80)
        logger.info('ETAPA 4/6: TRATAMENTO DE CORRELAÇÃO')
        logger.info('='*80)
        
        df_reduced = self.correlation_handler.fit_transform(df)
        
        # Salvar checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_03_correlation_treated.csv'
        save_data(df_reduced, checkpoint_path)
        logger.info(f'Checkpoint salvo: {checkpoint_path}')
        
        return df_reduced
    
    def run_outliers_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa tratamento de outliers.
        
        Args:
            df: DataFrame sem variáveis correlacionadas
        
        Returns:
            DataFrame sem outliers
        """
        logger.info('='*80)
        logger.info('ETAPA 5/6: TRATAMENTO DE OUTLIERS')
        logger.info('='*80)
        
        df_clean = self.outliers_handler.fit_transform(df)
        
        # Salvar checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_04_outliers_treated.csv'
        save_data(df_clean, checkpoint_path)
        logger.info(f'Checkpoint salvo: {checkpoint_path}')
        
        return df_clean
    
    def run_class_balancing(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executa balanceamento de classes.
        
        Args:
            df: DataFrame sem outliers
        
        Returns:
            Tupla (X_train_balanced, X_test, y_train_balanced, y_test)
        """
        logger.info('='*80)
        logger.info('ETAPA 6/6: BALANCEAMENTO DE CLASSES')
        logger.info('='*80)
        
        X_train, X_test, y_train, y_test = self.class_balancer.fit_transform(df)
        
        # Salvar conjuntos finais
        train_df = X_train.copy()
        train_df[self.config['target']['column_name']] = y_train
        train_path = self.output_dir / 'train_balanced.csv'
        save_data(train_df, train_path)
        logger.info(f'Conjunto de treino balanceado salvo: {train_path}')
        
        test_df = X_test.copy()
        test_df[self.config['target']['column_name']] = y_test
        test_path = self.output_dir / 'test.csv'
        save_data(test_df, test_path)
        logger.info(f'Conjunto de teste salvo: {test_path}')
        
        return X_train, X_test, y_train, y_test
    
    def run(
        self,
        input_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executa o pipeline completo de pré-processamento.
        
        Args:
            input_path: Caminho para os dados brutos (opcional)
        
        Returns:
            Tupla (X_train_balanced, X_test, y_train_balanced, y_test)
        """
        logger.info('='*80)
        logger.info('INICIANDO PIPELINE COMPLETO DE PRÉ-PROCESSAMENTO')
        logger.info('='*80)
        
        # Carregar dados
        if input_path is None:
            input_path = self.config['data']['raw_data_path']
        
        logger.info(f'Carregando dados de: {input_path}')
        df = load_data(input_path)
        logger.info(f'Dimensões iniciais: {df.shape}')
        
        # Executar etapas sequencialmente
        try:
            # Etapa 1: Análise Exploratória
            exploratory_results = self.run_exploratory_analysis(df)
            
            # Etapa 2: Tratamento de Valores Ausentes
            df = self.run_missing_values_treatment(df)
            
            # Etapa 3: Codificação Categórica
            df = self.run_categorical_encoding(df)
            
            # Etapa 4: Tratamento de Correlação
            df = self.run_correlation_treatment(df)
            
            # Etapa 5: Tratamento de Outliers
            df = self.run_outliers_treatment(df)
            
            # Etapa 6: Balanceamento de Classes
            X_train, X_test, y_train, y_test = self.run_class_balancing(df)
            
            logger.info('='*80)
            logger.info('PIPELINE COMPLETO DE PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO')
            logger.info('='*80)
            
            logger.info(f'Dimensões finais:')
            logger.info(f'  Treino: {X_train.shape}')
            logger.info(f'  Teste: {X_test.shape}')
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f'Erro durante execução do pipeline: {e}')
            raise
    
    def run_from_checkpoint(
        self,
        checkpoint_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executa o pipeline a partir de um checkpoint.
        
        Args:
            checkpoint_name: Nome do checkpoint ('imputed', 'encoded', etc.)
        
        Returns:
            Tupla (X_train_balanced, X_test, y_train_balanced, y_test)
        """
        checkpoint_map = {
            'imputed': ('checkpoint_01_imputed.csv', 3),
            'encoded': ('checkpoint_02_encoded.csv', 4),
            'correlation_treated': ('checkpoint_03_correlation_treated.csv', 5),
            'outliers_treated': ('checkpoint_04_outliers_treated.csv', 6)
        }
        
        if checkpoint_name not in checkpoint_map:
            raise ValueError(f'Checkpoint inválido: {checkpoint_name}')
        
        checkpoint_file, start_step = checkpoint_map[checkpoint_name]
        checkpoint_path = self.output_dir / checkpoint_file
        
        logger.info(f'Carregando checkpoint: {checkpoint_path}')
        df = load_data(checkpoint_path)
        
        logger.info(f'Retomando pipeline a partir da etapa {start_step}')
        
        # Executar etapas restantes
        if start_step <= 3:
            df = self.run_categorical_encoding(df)
        if start_step <= 4:
            df = self.run_correlation_treatment(df)
        if start_step <= 5:
            df = self.run_outliers_treatment(df)
        if start_step <= 6:
            X_train, X_test, y_train, y_test = self.run_class_balancing(df)
        
        return X_train, X_test, y_train, y_test


def main():
    """Função principal para execução standalone"""
    from src.utils import load_config
    
    config = load_config()
    pipeline = PreprocessingPipeline(config)
    
    # Executar pipeline completo
    X_train, X_test, y_train, y_test = pipeline.run()
    
    print('\n' + '='*80)
    print('✅ PIPELINE DE PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!')
    print('='*80)
    print(f'\n📊 Conjuntos de dados prontos:')
    print(f'   - Treino balanceado: data/processed/train_balanced.csv')
    print(f'   - Teste: data/processed/test.csv')
    print(f'\n📈 Dimensões finais:')
    print(f'   - Treino: {X_train.shape}')
    print(f'   - Teste: {X_test.shape}')
    print(f'\n📁 Checkpoints salvos em: data/processed/')
    print(f'📁 Resultados em: results/preprocessing/')
    print('\n' + '='*80)


if __name__ == '__main__':
    main()
