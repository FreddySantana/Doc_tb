"""
Pipeline de Pré-processamento Correto

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-15
Última Modificação: 2025-11-30

Descrição:
    Orquestra todas as etapas de pré-processamento na ordem CORRETA conforme
    descrito na Seção 4.2 da tese.

⚠️  ORDEM CRÍTICA DO PIPELINE (NÃO MUDAR!):
1. Carregamento de dados
2. Análise exploratória
3. Tratamento de valores ausentes (MICE + Moda)
4. Tratamento de outliers
5. Encoding de variáveis categóricas
6. Normalização/Padronização
7. Tratamento de correlação (VIF)
8. Split treino/teste (ANTES de SMOTE!)
9. Balanceamento de classes (SMOTE - APENAS no treino)

Licença: MIT
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import json

from src.data.data_loader import TBDataLoader
from src.preprocessing.missing_values import MissingValuesHandler
from src.preprocessing.outliers_treatment import OutliersHandler
from src.preprocessing.categorical_encoding import CategoricalEncoder
from src.preprocessing.class_balancing import ClassBalancer
from src.utils import setup_logger

logger = setup_logger(__name__)


class PreprocessingPipelineCorrected:
    """
    Pipeline CORRETO de pré-processamento com ordem apropriada.
    
    ⚠️  ORDEM CRÍTICA:
    1. Valores ausentes (ANTES de encoding)
    2. Outliers
    3. Encoding (ANTES de SMOTE)
    4. Split treino/teste (ANTES de SMOTE)
    5. SMOTE (APENAS no treino)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o pipeline de pré-processamento.
        
        Parâmetros:
        -----------
        config : Dict[str, Any]
            Dicionário de configurações
        """
        self.config = config
        self.output_dir = Path('results/preprocessing')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.data_loader = TBDataLoader(config['data']['path'])
        self.missing_handler = MissingValuesHandler(config)
        self.outliers_handler = OutliersHandler(config)
        self.encoder = CategoricalEncoder(config)
        self.class_balancer = ClassBalancer(config)
        
        logger.info('✅ PreprocessingPipelineCorrected inicializado')
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executa o pipeline CORRETO de pré-processamento.
        
        ⚠️  ORDEM CRÍTICA (NÃO MUDAR):
        1. Carregamento
        2. Análise exploratória
        3. Valores ausentes (MICE + Moda)
        4. Outliers
        5. Encoding (ANTES de SMOTE!)
        6. Split treino/teste (ANTES de SMOTE!)
        7. SMOTE (APENAS no treino!)
        
        Retorna:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            (X_train_balanced, X_test, y_train_balanced, y_test)
        """
        logger.info('='*80)
        logger.info('INICIANDO PIPELINE CORRETO DE PRÉ-PROCESSAMENTO')
        logger.info('='*80)
        
        # ETAPA 1: Carregamento de dados
        logger.info('\n[ETAPA 1/7] Carregando dados...')
        df = self.data_loader.load_data()
        logger.info(f'✅ Dados carregados: {df.shape[0]:,} linhas x {df.shape[1]} colunas')
        
        # ETAPA 2: Análise exploratória
        logger.info('\n[ETAPA 2/7] Análise exploratória...')
        self._exploratory_analysis(df)
        logger.info('✅ Análise exploratória concluída')
        
        # ETAPA 3: Tratamento de valores ausentes (ANTES de encoding!)
        logger.info('\n[ETAPA 3/7] Tratamento de valores ausentes (MICE + Moda)...')
        logger.info('⚠️  Imputando variáveis categóricas por MODA')
        logger.info('⚠️  Imputando variáveis numéricas por MICE')
        df = self.missing_handler.fit_transform(df, strategy='mice')
        logger.info(f'✅ Valores ausentes tratados')
        
        # ETAPA 4: Tratamento de outliers
        logger.info('\n[ETAPA 4/7] Tratamento de outliers (Isolation Forest)...')
        df = self.outliers_handler.fit_transform(df)
        logger.info(f'✅ Outliers tratados: {df.shape[0]:,} amostras restantes')
        
        # ETAPA 5: Encoding de variáveis categóricas (ANTES de SMOTE!)
        logger.info('\n[ETAPA 5/7] Encoding de variáveis categóricas...')
        logger.info('⚠️  One-Hot para ≤5 categorias, Label para >5 categorias')
        df = self.encoder.fit_transform(df, strategy='mixed')
        logger.info(f'✅ Encoding concluído: {df.shape[1]} colunas')
        
        # ETAPA 6: Split treino/teste (ANTES de SMOTE!)
        logger.info('\n[ETAPA 6/7] Split treino/teste (80/20 estratificado)...')
        logger.info('⚠️  IMPORTANTE: Split ANTES de SMOTE para evitar data leakage!')
        
        # ETAPA 7: Balanceamento de classes (SMOTE - APENAS no treino)
        logger.info('\n[ETAPA 7/7] Balanceamento de classes (SMOTE)...')
        logger.info('⚠️  IMPORTANTE: SMOTE aplicado APENAS no conjunto de treino!')
        logger.info('⚠️  Conjunto de teste NÃO é balanceado (reflete distribuição real)')
        X_train_balanced, X_test, y_train_balanced, y_test = self.class_balancer.fit_transform(df)
        
        logger.info('='*80)
        logger.info('✅ PIPELINE CORRETO DE PRÉ-PROCESSAMENTO CONCLUÍDO!')
        logger.info('='*80)
        
        # Gerar relatório final
        self._generate_final_report(
            df, X_train_balanced, X_test, y_train_balanced, y_test
        )
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def _exploratory_analysis(self, df: pd.DataFrame) -> None:
        """
        Realiza análise exploratória dos dados.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
        """
        logger.info('Realizando análise exploratória...')
        
        # Dimensões
        logger.info(f'  Dimensões: {df.shape[0]:,} linhas x {df.shape[1]} colunas')
        
        # Tipos de dados
        dtypes = df.dtypes.value_counts()
        logger.info(f'  Tipos de dados:')
        for dtype, count in dtypes.items():
            logger.info(f'    - {dtype}: {count}')
        
        # Valores ausentes
        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / (df.shape[0] * df.shape[1]) * 100)
        logger.info(f'  Valores ausentes: {missing_total:,} ({missing_pct:.2f}%)')
        
        # Target
        target_col = self.config['target']['column_name']
        if target_col in df.columns:
            logger.info(f'  Distribuição do target ({target_col}):')
            for value, count in df[target_col].value_counts().items():
                pct = (count / len(df) * 100)
                logger.info(f'    - {value}: {count:,} ({pct:.2f}%)')
    
    def _generate_final_report(
        self,
        df_original: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """
        Gera relatório final do pré-processamento.
        
        Parâmetros:
        -----------
        df_original : pd.DataFrame
            DataFrame original
        X_train : pd.DataFrame
            Features de treino
        X_test : pd.DataFrame
            Features de teste
        y_train : pd.Series
            Target de treino
        y_test : pd.Series
            Target de teste
        """
        logger.info('\n📊 RELATÓRIO FINAL DO PRÉ-PROCESSAMENTO:')
        logger.info('='*80)
        
        report = {
            'original': {
                'shape': list(df_original.shape),
                'columns': len(df_original.columns)
            },
            'train': {
                'X_shape': list(X_train.shape),
                'y_shape': list(y_train.shape),
                'y_distribution': y_train.value_counts().to_dict()
            },
            'test': {
                'X_shape': list(X_test.shape),
                'y_shape': list(y_test.shape),
                'y_distribution': y_test.value_counts().to_dict()
            }
        }
        
        logger.info(f'Dataset Original: {df_original.shape[0]:,} x {df_original.shape[1]}')
        logger.info(f'Treino (com SMOTE): {X_train.shape[0]:,} x {X_train.shape[1]}')
        logger.info(f'Teste (sem SMOTE): {X_test.shape[0]:,} x {X_test.shape[1]}')
        logger.info(f'Total de features: {X_train.shape[1]}')
        
        logger.info(f'\nDistribuição do target:')
        logger.info(f'  Treino: {dict(y_train.value_counts())}')
        logger.info(f'  Teste: {dict(y_test.value_counts())}')
        
        # Salvar relatório
        report_path = self.output_dir / 'preprocessing_report_final.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f'\n✅ Relatório salvo em {report_path}')


if __name__ == "__main__":
    logger.info("Preprocessing Pipeline Corrected - Exemplo de uso")
