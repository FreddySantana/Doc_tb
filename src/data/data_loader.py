"""
Módulo para carregamento de dados reais de tuberculose

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-15
Última Modificação: 2025-11-20

Descrição:
    Este módulo fornece funcionalidades para carregar, validar e preparar
    o dataset real de tuberculose (2006-2016) com 103.846 pacientes.
    
    Funcionalidades principais:
    - Carregamento do CSV
    - Validação de integridade
    - Separação de features e target
    - Estatísticas descritivas
    - Preparação para modelagem

Dependências:
    - pandas
    - numpy
    - logging

Uso:
    >>> from src.data.data_loader import TBDataLoader
    >>> loader = TBDataLoader('data/tuberculosis-data-06-16.csv')
    >>> df = loader.load_data()
    >>> X, y = loader.get_features_target(df)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TBDataLoader:
    """
    Classe para carregamento e preparação de dados reais de tuberculose.
    
    Attributes:
        data_path (str): Caminho para o arquivo CSV
        target_column (str): Nome da coluna target (desfecho)
        df (pd.DataFrame): DataFrame carregado
    """
    
    def __init__(self, data_path: str, target_column: str = 'sitAtual'):
        """
        Inicializa o carregador de dados.
        
        Args:
            data_path: Caminho para o arquivo CSV
            target_column: Nome da coluna target (default: 'sitAtual')
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.df = None
        
        # Validar se arquivo existe
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {self.data_path}")
        
        logger.info(f"TBDataLoader inicializado com dataset: {self.data_path}")
    
    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Carrega dados do CSV.
        
        Args:
            nrows: Número de linhas a carregar (None = todas)
        
        Returns:
            DataFrame com os dados carregados
        """
        logger.info("="*70)
        logger.info("CARREGANDO DADOS REAIS DE TUBERCULOSE")
        logger.info("="*70)
        
        try:
            # Carregar CSV
            self.df = pd.read_csv(
                self.data_path,
                nrows=nrows,
                low_memory=False
            )
            
            # Estatísticas básicas
            n_rows, n_cols = self.df.shape
            logger.info(f"✓ Dados carregados com sucesso!")
            logger.info(f"  - Pacientes: {n_rows:,}")
            logger.info(f"  - Features: {n_cols}")
            logger.info(f"  - Período: 2006-2016")
            
            # Validar coluna target
            if self.target_column not in self.df.columns:
                raise ValueError(f"Coluna target '{self.target_column}' não encontrada")
            
            # Distribuição do target
            target_dist = self.df[self.target_column].value_counts()
            logger.info(f"\n  Distribuição do Target ({self.target_column}):")
            for value, count in target_dist.items():
                pct = (count / n_rows) * 100
                logger.info(f"    - {value}: {count:,} ({pct:.1f}%)")
            
            # Valores ausentes
            missing = self.df.isnull().sum().sum()
            missing_pct = (missing / (n_rows * n_cols)) * 100
            logger.info(f"\n  Valores Ausentes:")
            logger.info(f"    - Total: {missing:,} ({missing_pct:.2f}%)")
            
            logger.info("="*70)
            
            return self.df
        
        except Exception as e:
            logger.error(f"✗ Erro ao carregar dados: {str(e)}")
            raise
    
    def get_features_target(
        self, 
        df: Optional[pd.DataFrame] = None,
        drop_columns: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features (X) e target (y).
        
        Args:
            df: DataFrame (usa self.df se None)
            drop_columns: Colunas adicionais para remover
        
        Returns:
            Tuple (X, y) com features e target
        """
        if df is None:
            if self.df is None:
                raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            df = self.df
        
        logger.info("Separando features e target...")
        
        # Colunas a remover
        cols_to_drop = [self.target_column]
        if drop_columns:
            cols_to_drop.extend(drop_columns)
        
        # Remover colunas que não existem
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        # Separar X e y
        X = df.drop(columns=cols_to_drop)
        y = df[self.target_column]
        
        logger.info(f"✓ Features: {X.shape[1]} colunas")
        logger.info(f"✓ Target: {y.shape[0]} amostras")
        
        return X, y
    
    def get_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Retorna estatísticas descritivas dos dados.
        
        Args:
            df: DataFrame (usa self.df se None)
        
        Returns:
            Dicionário com estatísticas
        """
        if df is None:
            if self.df is None:
                raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            df = self.df
        
        stats = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,  # -1 para target
            'target_column': self.target_column,
            'target_distribution': df[self.target_column].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return stats
    
    def prepare_for_ml(
        self,
        df: Optional[pd.DataFrame] = None,
        encode_target: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dados para modelagem de ML.
        
        Args:
            df: DataFrame (usa self.df se None)
            encode_target: Se True, codifica target como 0/1
        
        Returns:
            Tuple (X, y) preparados para ML
        """
        if df is None:
            if self.df is None:
                raise ValueError("Dados não carregados. Execute load_data() primeiro.")
            df = self.df.copy()
        else:
            df = df.copy()
        
        logger.info("Preparando dados para ML...")
        
        # Separar X e y
        X, y = self.get_features_target(df)
        
        # Identificar colunas categóricas
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            logger.info(f"Codificando {len(categorical_cols)} variáveis categóricas...")
            
            # One-Hot Encoding para variáveis categóricas
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
            
            logger.info(f"✓ Features após encoding: {X.shape[1]} colunas")
        
        # Tratar valores ausentes
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Tratando {missing_count:,} valores ausentes...")
            
            # Imputar com mediana para numéricas
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            
            # Preencher restantes (se houver) com 0
            X = X.fillna(0)
            
            logger.info(f"✓ Valores ausentes tratados")
        
        # Codificar target se necessário
        if encode_target:
            # Abandono = 1, Cura = 0
            y_encoded = (y == 'Abandono').astype(int)
            logger.info(f"✓ Target codificado: Abandono=1 ({y_encoded.sum()}), Cura=0 ({(~y_encoded.astype(bool)).sum()})")
            y = y_encoded
        
        return X, y
    
    def get_column_names(self) -> Dict[str, list]:
        """
        Retorna nomes das colunas por tipo.
        
        Returns:
            Dicionário com listas de colunas por tipo
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remover target das listas
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'target': self.target_column,
            'all_features': numeric_cols + categorical_cols
        }


def load_tb_data(
    data_path: str = 'data/tuberculosis-data-06-16.csv',
    prepare_ml: bool = True,
    nrows: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Função auxiliar para carregar dados de TB rapidamente.
    
    Args:
        data_path: Caminho para o CSV
        prepare_ml: Se True, prepara para ML (codifica target)
        nrows: Número de linhas (None = todas)
    
    Returns:
        Tuple (X, y) com dados preparados
    
    Example:
        >>> X, y = load_tb_data()
        >>> print(f"Dados: {X.shape}, Target: {y.shape}")
    """
    loader = TBDataLoader(data_path)
    df = loader.load_data(nrows=nrows)
    
    if prepare_ml:
        X, y = loader.prepare_for_ml(df)
    else:
        X, y = loader.get_features_target(df)
    
    return X, y


if __name__ == "__main__":
    # Teste do módulo
    print("="*70)
    print("TESTE DO MÓDULO DATA_LOADER")
    print("="*70)
    
    # Carregar dados
    X, y = load_tb_data()
    
    print(f"\n✓ Dados carregados com sucesso!")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - y distribution: {y.value_counts().to_dict()}")
    
    print("\n✓ Módulo funcionando corretamente!")
    print("="*70)
