"""
Módulo para Encoding de Variáveis Categóricas

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-15
Última Modificação: 2025-11-30

Descrição:
    Implementa estratégias de encoding para variáveis categóricas conforme
    descrito na Seção 4.2 da tese (Pré-processamento e Qualidade de Dados).
    
    Estratégias:
    - Label Encoding para variáveis ordinais
    - One-Hot Encoding para variáveis nominais
    - Frequency Encoding como alternativa

Licença: MIT
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils import setup_logger

logger = setup_logger(__name__)


class CategoricalEncoder:
    """
    Classe para encoding de variáveis categóricas.
    
    Implementa diferentes estratégias de encoding para preparar
    dados categóricos para algoritmos de ML.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o encoder de variáveis categóricas.
        
        Parâmetros:
        -----------
        config : Dict[str, Any]
            Dicionário de configurações
        """
        self.config = config
        self.output_dir = Path('results/preprocessing/encoding')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dicionários para armazenar encoders
        self.label_encoders = {}
        self.one_hot_encoder = None
        self.categorical_features = []
        self.numerical_features = []
        
        logger.info("✅ CategoricalEncoder inicializado")
    
    def identify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identifica variáveis categóricas e numéricas.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
            
        Retorna:
        --------
        Tuple[List[str], List[str]]
            (variáveis_categóricas, variáveis_numéricas)
        """
        logger.info("Identificando variáveis categóricas e numéricas...")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        self.categorical_features = categorical_cols
        self.numerical_features = numerical_cols
        
        logger.info(f"  Variáveis categóricas: {len(categorical_cols)}")
        logger.info(f"  Variáveis numéricas: {len(numerical_cols)}")
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            logger.info(f"    - {col}: {n_unique} categorias únicas")
        
        return categorical_cols, numerical_cols
    
    def analyze_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa características das variáveis categóricas.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
            
        Retorna:
        --------
        pd.DataFrame
            Análise das variáveis categóricas
        """
        logger.info("Analisando variáveis categóricas...")
        
        categorical_cols, _ = self.identify_features(df)
        
        analysis = []
        for col in categorical_cols:
            analysis.append({
                'Variável': col,
                'Tipo': 'Categórica',
                'Valores Únicos': df[col].nunique(),
                'Valores Ausentes (%)': (df[col].isnull().sum() / len(df) * 100),
                'Valor Mais Frequente': df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A',
                'Frequência (%)': (df[col].value_counts().iloc[0] / len(df) * 100) if len(df[col].value_counts()) > 0 else 0
            })
        
        analysis_df = pd.DataFrame(analysis)
        return analysis_df
    
    def apply_label_encoding(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aplica Label Encoding para variáveis ordinais.
        
        Label Encoding é apropriado para variáveis ordinais onde
        existe uma ordem natural entre as categorias.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
        columns : Optional[List[str]]
            Colunas para aplicar Label Encoding
            
        Retorna:
        --------
        pd.DataFrame
            DataFrame com Label Encoding aplicado
        """
        logger.info("Aplicando Label Encoding...")
        
        df_encoded = df.copy()
        
        if columns is None:
            categorical_cols, _ = self.identify_features(df)
            columns = categorical_cols
        
        for col in columns:
            if col in df_encoded.columns:
                logger.info(f"  Label Encoding em '{col}'")
                
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        logger.info("✅ Label Encoding concluído")
        return df_encoded
    
    def apply_one_hot_encoding(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Aplica One-Hot Encoding para variáveis nominais.
        
        One-Hot Encoding é apropriado para variáveis nominais onde
        não existe ordem entre as categorias.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
        columns : Optional[List[str]]
            Colunas para aplicar One-Hot Encoding
        drop_first : bool
            Se deve descartar a primeira coluna (evita multicolinearidade)
            
        Retorna:
        --------
        pd.DataFrame
            DataFrame com One-Hot Encoding aplicado
        """
        logger.info("Aplicando One-Hot Encoding...")
        
        if columns is None:
            categorical_cols, _ = self.identify_features(df)
            columns = categorical_cols
        
        # Aplicar One-Hot Encoding
        df_encoded = pd.get_dummies(
            df,
            columns=columns,
            drop_first=drop_first,
            dtype='int64'
        )
        
        logger.info(f"  Colunas antes: {df.shape[1]}")
        logger.info(f"  Colunas depois: {df_encoded.shape[1]}")
        logger.info("✅ One-Hot Encoding concluído")
        
        return df_encoded
    
    def apply_frequency_encoding(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aplica Frequency Encoding para variáveis categóricas.
        
        Frequency Encoding substitui cada categoria pela frequência
        relativa de ocorrência.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
        columns : Optional[List[str]]
            Colunas para aplicar Frequency Encoding
            
        Retorna:
        --------
        pd.DataFrame
            DataFrame com Frequency Encoding aplicado
        """
        logger.info("Aplicando Frequency Encoding...")
        
        df_encoded = df.copy()
        
        if columns is None:
            categorical_cols, _ = self.identify_features(df)
            columns = categorical_cols
        
        for col in columns:
            if col in df_encoded.columns:
                logger.info(f"  Frequency Encoding em '{col}'")
                
                # Calcular frequência relativa
                freq_map = df_encoded[col].value_counts(normalize=True).to_dict()
                df_encoded[col] = df_encoded[col].map(freq_map)
        
        logger.info("✅ Frequency Encoding concluído")
        return df_encoded
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        strategy: str = 'mixed'
    ) -> pd.DataFrame:
        """
        Aplica encoding de variáveis categóricas.
        
        Estratégias:
        - 'label': Label Encoding para todas
        - 'onehot': One-Hot Encoding para todas
        - 'frequency': Frequency Encoding para todas
        - 'mixed': Análise automática e aplicação apropriada
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados
        strategy : str
            Estratégia de encoding
            
        Retorna:
        --------
        pd.DataFrame
            DataFrame com encoding aplicado
        """
        logger.info("="*80)
        logger.info("INICIANDO ENCODING DE VARIÁVEIS CATEGÓRICAS")
        logger.info("="*80)
        
        # Analisar variáveis
        categorical_cols, numerical_cols = self.identify_features(df)
        
        if not categorical_cols:
            logger.info("Nenhuma variável categórica encontrada")
            return df
        
        # Aplicar estratégia
        if strategy == 'label':
            df_encoded = self.apply_label_encoding(df, categorical_cols)
        elif strategy == 'onehot':
            df_encoded = self.apply_one_hot_encoding(df, categorical_cols)
        elif strategy == 'frequency':
            df_encoded = self.apply_frequency_encoding(df, categorical_cols)
        elif strategy == 'mixed':
            # Estratégia mista: One-Hot para poucas categorias, Label para muitas
            df_encoded = df.copy()
            
            for col in categorical_cols:
                n_unique = df[col].nunique()
                
                if n_unique <= 5:
                    # One-Hot para poucas categorias
                    logger.info(f"  One-Hot Encoding em '{col}' ({n_unique} categorias)")
                    df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True, dtype='int64')
                else:
                    # Label Encoding para muitas categorias
                    logger.info(f"  Label Encoding em '{col}' ({n_unique} categorias)")
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
        else:
            raise ValueError(f"Estratégia desconhecida: {strategy}")
        
        logger.info("="*80)
        logger.info("ENCODING DE VARIÁVEIS CATEGÓRICAS CONCLUÍDO")
        logger.info("="*80)
        
        return df_encoded


if __name__ == "__main__":
    logger.info("Categorical Encoding Module - Exemplo de uso")
