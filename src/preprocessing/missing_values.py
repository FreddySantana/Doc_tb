"""
Módulo para Tratamento de Valores Ausentes

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-15
Última Modificação: 2025-11-30

Descrição:
    Implementa estratégias de imputação conforme descrito na Seção 4.2 da tese:
    - MICE (Multivariate Imputation by Chained Equations) para variáveis numéricas
    - Imputação por moda para variáveis categóricas
    - KNN Imputer como estratégia alternativa

Licença: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.utils import setup_logger

logger = setup_logger(__name__)


class MissingValuesHandler:
    """
    Classe para tratamento de valores ausentes em dados de tuberculose.
    
    Implementa as estratégias descritas na tese:
    1. MICE (Multivariate Imputation by Chained Equations) para numéricas
    2. Moda para variáveis categóricas
    3. KNN Imputer como alternativa
    
    IMPORTANTE: Deve ser aplicado ANTES de qualquer encoding!
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o handler de valores ausentes.
        
        Parâmetros:
        -----------
        config : Dict[str, Any]
            Dicionário de configurações
        """
        self.config = config
        self.strategy = config.get('preprocessing', {}).get('missing_values_strategy', 'mice')
        self.output_dir = Path('results/preprocessing/missing_values')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar imputers
        self.mice_imputer = IterativeImputer(
            max_iter=10,
            random_state=config.get('random_state', 42),
            verbose=0
        )
        
        self.knn_imputer = KNNImputer(
            n_neighbors=5,
            weights='uniform'
        )
        
        logger.info(f'✅ MissingValuesHandler inicializado com estratégia: {self.strategy}')
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa padrões de valores ausentes no dataset.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com estatísticas de valores ausentes
        """
        logger.info('Analisando valores ausentes...')
        
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Variável': missing_counts.index,
            'Valores Ausentes': missing_counts.values,
            'Percentual (%)': missing_percentages.values
        })
        
        # Filtrar apenas variáveis com valores ausentes
        missing_df = missing_df[missing_df['Valores Ausentes'] > 0]
        missing_df = missing_df.sort_values('Percentual (%)', ascending=False)
        missing_df = missing_df.reset_index(drop=True)
        
        logger.info(f'  Total de variáveis com valores ausentes: {len(missing_df)}')
        logger.info(f'  Total de valores ausentes: {missing_counts.sum():,}')
        
        return missing_df
    
    def plot_missing_values(
        self,
        df: pd.DataFrame,
        missing_df: pd.DataFrame,
        suffix: str = 'before'
    ) -> None:
        """
        Gera visualizações de valores ausentes.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
        missing_df : pd.DataFrame
            DataFrame com estatísticas de valores ausentes
        suffix : str
            Sufixo para nome do arquivo ('before' ou 'after')
        """
        logger.info(f'Gerando visualizações de valores ausentes ({suffix})')
        
        # Heatmap de valores ausentes (sample)
        if len(df) > 1000:
            df_sample = df.sample(n=1000, random_state=42)
        else:
            df_sample = df
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(df_sample.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title(f'Mapa de Valores Ausentes ({suffix.title()})', fontsize=14, fontweight='bold')
        plt.xlabel('Variáveis', fontsize=12)
        plt.tight_layout()
        
        output_path = self.output_dir / f'missing_values_heatmap_{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'  Heatmap salvo em {output_path}')
        
        # Gráfico de barras (top 20)
        if not missing_df.empty:
            top_missing = missing_df.head(20)
            
            plt.figure(figsize=(12, 8))
            plt.barh(top_missing['Variável'], top_missing['Percentual (%)'], color='coral')
            plt.xlabel('Percentual de Valores Ausentes (%)', fontsize=12)
            plt.ylabel('Variável', fontsize=12)
            plt.title(f'Top 20 Variáveis com Valores Ausentes ({suffix.title()})', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            output_path = self.output_dir / f'missing_values_barplot_{suffix}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f'  Gráfico de barras salvo em {output_path}')
    
    def impute_mice(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica imputação MICE para variáveis numéricas.
        
        IMPORTANTE: MICE só funciona com dados numéricos!
        Variáveis categóricas devem ser imputadas por moda ANTES.
        
        Conforme descrito na tese, MICE é uma técnica avançada que:
        - Modela cada variável com valores ausentes como função das outras
        - Itera até convergência
        - Captura relações multivariadas complexas
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com valores ausentes (APENAS NUMÉRICAS)
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com valores imputados
        """
        logger.info('Aplicando imputação MICE para variáveis numéricas...')
        
        df_imputed = df.copy()
        
        # Separar variáveis categóricas e numéricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        logger.info(f'  Variáveis categóricas: {len(categorical_cols)}')
        logger.info(f'  Variáveis numéricas: {len(numeric_cols)}')
        
        # PASSO 1: Imputar categóricas por moda PRIMEIRO
        if categorical_cols:
            logger.info('  [PASSO 1] Imputando variáveis categóricas por moda...')
            for col in categorical_cols:
                if df_imputed[col].isnull().sum() > 0:
                    mode_value = df_imputed[col].mode()
                    if len(mode_value) > 0:
                        df_imputed[col].fillna(mode_value[0], inplace=True)
                    else:
                        df_imputed[col].fillna('Desconhecido', inplace=True)
            logger.info('  ✅ Imputação por moda concluída')
        
        # PASSO 2: Aplicar MICE para variáveis numéricas
        if numeric_cols:
            logger.info('  [PASSO 2] Aplicando MICE para variáveis numéricas...')
            numeric_data = df_imputed[numeric_cols]
            numeric_imputed = self.mice_imputer.fit_transform(numeric_data)
            df_imputed[numeric_cols] = numeric_imputed
            logger.info('  ✅ MICE concluído para variáveis numéricas')
        
        # Verificar valores ausentes restantes
        remaining_missing = df_imputed.isnull().sum().sum()
        logger.info(f'  Valores ausentes restantes: {remaining_missing}')
        
        return df_imputed
    
    def impute_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica imputação KNN (K-Nearest Neighbors).
        
        IMPORTANTE: KNN também só funciona com dados numéricos!
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com valores ausentes
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com valores imputados
        """
        logger.info('Aplicando imputação KNN...')
        
        df_imputed = df.copy()
        
        # Separar variáveis categóricas e numéricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # PASSO 1: Imputar categóricas por moda PRIMEIRO
        if categorical_cols:
            logger.info('  [PASSO 1] Imputando variáveis categóricas por moda...')
            for col in categorical_cols:
                if df_imputed[col].isnull().sum() > 0:
                    mode_value = df_imputed[col].mode()
                    if len(mode_value) > 0:
                        df_imputed[col].fillna(mode_value[0], inplace=True)
                    else:
                        df_imputed[col].fillna('Desconhecido', inplace=True)
            logger.info('  ✅ Imputação por moda concluída')
        
        # PASSO 2: Aplicar KNN para variáveis numéricas
        if numeric_cols:
            logger.info('  [PASSO 2] Aplicando KNN para variáveis numéricas...')
            numeric_data = df_imputed[numeric_cols]
            numeric_imputed = self.knn_imputer.fit_transform(numeric_data)
            df_imputed[numeric_cols] = numeric_imputed
            logger.info('  ✅ KNN concluído para variáveis numéricas')
        
        remaining_missing = df_imputed.isnull().sum().sum()
        logger.info(f'  Valores ausentes restantes: {remaining_missing}')
        
        return df_imputed
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aplica a estratégia de imputação selecionada.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com valores ausentes
        strategy : Optional[str]
            Estratégia de imputação ('mice', 'knn')
            Se None, usa a estratégia configurada
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com valores imputados
        """
        logger.info('='*80)
        logger.info('INICIANDO TRATAMENTO DE VALORES AUSENTES')
        logger.info('='*80)
        
        # Analisar valores ausentes antes
        missing_before = self.analyze_missing_values(df)
        self.plot_missing_values(df, missing_before, suffix='before')
        
        # Selecionar estratégia
        if strategy is None:
            strategy = self.strategy
        
        # Aplicar imputação
        if strategy == 'mice':
            df_imputed = self.impute_mice(df)
        elif strategy == 'knn':
            df_imputed = self.impute_knn(df)
        else:
            raise ValueError(f"Estratégia desconhecida: {strategy}")
        
        # Analisar valores ausentes depois
        missing_after = self.analyze_missing_values(df_imputed)
        self.plot_missing_values(df_imputed, missing_after, suffix='after')
        
        logger.info('='*80)
        logger.info('TRATAMENTO DE VALORES AUSENTES CONCLUÍDO')
        logger.info('='*80)
        
        return df_imputed


if __name__ == "__main__":
    logger.info("Missing Values Module - Exemplo de uso")
