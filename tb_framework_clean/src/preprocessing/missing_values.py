"""
Módulo para tratamento de valores ausentes no dataset

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-15
Última Modificação: 2025-03-10

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
Tratamento de Valores Ausentes

Implementa estratégias de imputação conforme descrito na Seção 4.2 da tese:
- MICE (Multivariate Imputation by Chained Equations) para variáveis numéricas
- Imputação por moda para variáveis categóricas
- KNN Imputer como estratégia alternativa

Referência: Seção 4.2 da tese - Pré-processamento e Qualidade de Dados
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
    1. MICE (Multivariate Imputation by Chained Equations)
    2. KNN Imputer
    3. Imputação por moda para categóricas
    
    Attributes:
        config: Dicionário de configurações
        strategy: Estratégia de imputação ('mice', 'knn', 'hybrid')
        imputer: Objeto imputer configurado
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o handler de valores ausentes.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        self.strategy = config.get('preprocessing', {}).get('missing_values_strategy', 'mice')
        self.output_dir = Path('results/preprocessing/missing_values')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar imputers
        self.mice_imputer = IterativeImputer(
            max_iter=10,
            random_state=config.get('random_state', 42)
        )
        
        self.knn_imputer = KNNImputer(
            n_neighbors=5,
            weights='uniform'
        )
        
        logger.info(f'MissingValuesHandler inicializado com estratégia: {self.strategy}')
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa padrões de valores ausentes no dataset.
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            DataFrame com estatísticas de valores ausentes
        """
        logger.info('Analisando valores ausentes')
        
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
        
        logger.info(f'Total de variáveis com valores ausentes: {len(missing_df)}')
        logger.info(f'Total de valores ausentes: {missing_counts.sum():,}')
        
        return missing_df
    
    def plot_missing_values(
        self,
        df: pd.DataFrame,
        missing_df: pd.DataFrame,
        suffix: str = 'before'
    ) -> None:
        """
        Gera visualizações de valores ausentes.
        
        Args:
            df: DataFrame com os dados
            missing_df: DataFrame com estatísticas de valores ausentes
            suffix: Sufixo para nome do arquivo ('before' ou 'after')
        """
        logger.info(f'Gerando visualizações de valores ausentes ({suffix})')
        
        # Heatmap de valores ausentes
        plt.figure(figsize=(14, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title(f'Mapa de Valores Ausentes ({suffix.title()})', fontsize=14, fontweight='bold')
        plt.xlabel('Variáveis', fontsize=12)
        plt.tight_layout()
        
        output_path = self.output_dir / f'missing_values_heatmap_{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'Heatmap salvo em {output_path}')
        
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
            
            logger.info(f'Gráfico de barras salvo em {output_path}')
    
    def impute_mice(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica imputação MICE (Multivariate Imputation by Chained Equations).
        
        Conforme descrito na tese, MICE é uma técnica avançada que:
        - Modela cada variável com valores ausentes como função das outras
        - Itera até convergência
        - Captura relações multivariadas complexas
        
        Args:
            df: DataFrame com valores ausentes
        
        Returns:
            DataFrame com valores imputados
        """
        logger.info('Aplicando imputação MICE')
        
        df_imputed = df.copy()
        
        # Separar variáveis categóricas e numéricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        logger.info(f'Variáveis categóricas: {len(categorical_cols)}')
        logger.info(f'Variáveis numéricas: {len(numeric_cols)}')
        
        # Aplicar MICE para variáveis numéricas
        if numeric_cols:
            logger.info('Aplicando MICE para variáveis numéricas')
            numeric_data = df_imputed[numeric_cols]
            numeric_imputed = self.mice_imputer.fit_transform(numeric_data)
            df_imputed[numeric_cols] = numeric_imputed
            logger.info('MICE concluído para variáveis numéricas')
        
        # Imputar categóricas por moda
        if categorical_cols:
            logger.info('Aplicando imputação por moda para variáveis categóricas')
            for col in categorical_cols:
                if df_imputed[col].isnull().sum() > 0:
                    mode_value = df_imputed[col].mode()
                    if len(mode_value) > 0:
                        df_imputed[col].fillna(mode_value[0], inplace=True)
                    else:
                        # Se não houver moda, usar 'Desconhecido'
                        df_imputed[col].fillna('Desconhecido', inplace=True)
            logger.info('Imputação por moda concluída')
        
        # Verificar valores ausentes restantes
        remaining_missing = df_imputed.isnull().sum().sum()
        logger.info(f'Valores ausentes restantes após MICE: {remaining_missing}')
        
        return df_imputed
    
    def impute_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica imputação KNN (K-Nearest Neighbors).
        
        Args:
            df: DataFrame com valores ausentes
        
        Returns:
            DataFrame com valores imputados
        """
        logger.info('Aplicando imputação KNN')
        
        df_imputed = df.copy()
        
        # Separar variáveis categóricas e numéricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Aplicar KNN para variáveis numéricas
        if numeric_cols:
            logger.info('Aplicando KNN para variáveis numéricas')
            numeric_data = df_imputed[numeric_cols]
            numeric_imputed = self.knn_imputer.fit_transform(numeric_data)
            df_imputed[numeric_cols] = numeric_imputed
            logger.info('KNN concluído para variáveis numéricas')
        
        # Imputar categóricas por moda
        if categorical_cols:
            logger.info('Aplicando imputação por moda para variáveis categóricas')
            for col in categorical_cols:
                if df_imputed[col].isnull().sum() > 0:
                    mode_value = df_imputed[col].mode()
                    if len(mode_value) > 0:
                        df_imputed[col].fillna(mode_value[0], inplace=True)
                    else:
                        df_imputed[col].fillna('Desconhecido', inplace=True)
            logger.info('Imputação por moda concluída')
        
        remaining_missing = df_imputed.isnull().sum().sum()
        logger.info(f'Valores ausentes restantes após KNN: {remaining_missing}')
        
        return df_imputed
    
    def impute_hybrid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica estratégia híbrida: KNN + Moda.
        
        Esta é a estratégia recomendada que combina:
        - KNN para variáveis numéricas (preserva distribuições)
        - Moda para variáveis categóricas (preserva categorias)
        
        Args:
            df: DataFrame com valores ausentes
        
        Returns:
            DataFrame com valores imputados
        """
        logger.info('Aplicando estratégia híbrida (KNN + Moda)')
        return self.impute_knn(df)
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aplica a estratégia de imputação selecionada.
        
        Args:
            df: DataFrame com valores ausentes
            strategy: Estratégia de imputação ('mice', 'knn', 'hybrid')
                     Se None, usa a estratégia configurada
        
        Returns:
            DataFrame com valores imputados
        """
        logger.info('='*80)
        logger.info('INICIANDO TRATAMENTO DE VALORES AUSENTES')
        logger.info('='*80)
        
        # Analisar valores ausentes antes
        missing_before = self.analyze_missing_values(df)
        self.plot_missing_values(df, missing_before, suffix='before')
        
        # Aplicar estratégia selecionada
        strategy = strategy or self.strategy
        
        if strategy == 'mice':
            df_imputed = self.impute_mice(df)
        elif strategy == 'knn':
            df_imputed = self.impute_knn(df)
        elif strategy == 'hybrid':
            df_imputed = self.impute_hybrid(df)
        else:
            raise ValueError(f'Estratégia inválida: {strategy}')
        
        # Analisar valores ausentes depois
        missing_after = self.analyze_missing_values(df_imputed)
        self.plot_missing_values(df_imputed, missing_after, suffix='after')
        
        # Gerar relatório
        self._generate_report(missing_before, missing_after, strategy)
        
        logger.info('='*80)
        logger.info('TRATAMENTO DE VALORES AUSENTES CONCLUÍDO')
        logger.info('='*80)
        
        return df_imputed
    
    def _generate_report(
        self,
        missing_before: pd.DataFrame,
        missing_after: pd.DataFrame,
        strategy: str
    ) -> None:
        """
        Gera relatório do tratamento de valores ausentes.
        
        Args:
            missing_before: Estatísticas antes da imputação
            missing_after: Estatísticas depois da imputação
            strategy: Estratégia utilizada
        """
        logger.info('Gerando relatório de tratamento de valores ausentes')
        
        report_path = self.output_dir / 'missing_values_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('# Relatório de Tratamento de Valores Ausentes\n\n')
            
            f.write(f'## Estratégia Utilizada: {strategy.upper()}\n\n')
            
            f.write('## Valores Ausentes ANTES da Imputação\n\n')
            if not missing_before.empty:
                f.write(f'- **Total de variáveis com valores ausentes:** {len(missing_before)}\n')
                f.write(f'- **Total de valores ausentes:** {missing_before["Valores Ausentes"].sum():,}\n\n')
                f.write('### Top 10 Variáveis\n\n')
                f.write(missing_before.head(10).to_markdown(index=False))
                f.write('\n\n')
            else:
                f.write('Nenhum valor ausente encontrado.\n\n')
            
            f.write('## Valores Ausentes DEPOIS da Imputação\n\n')
            if not missing_after.empty:
                f.write(f'- **Total de variáveis com valores ausentes:** {len(missing_after)}\n')
                f.write(f'- **Total de valores ausentes:** {missing_after["Valores Ausentes"].sum():,}\n\n')
            else:
                f.write('✅ **Todos os valores ausentes foram tratados com sucesso!**\n\n')
            
            f.write('## Descrição da Estratégia\n\n')
            if strategy == 'mice':
                f.write('**MICE (Multivariate Imputation by Chained Equations)**\n\n')
                f.write('- Modela cada variável com valores ausentes como função das outras\n')
                f.write('- Itera até convergência\n')
                f.write('- Captura relações multivariadas complexas\n')
                f.write('- Recomendado pela tese para dados de tuberculose\n')
            elif strategy == 'knn':
                f.write('**KNN (K-Nearest Neighbors)**\n\n')
                f.write('- Imputa valores baseado nos k vizinhos mais próximos\n')
                f.write('- Preserva distribuições locais\n')
                f.write('- Eficiente computacionalmente\n')
            elif strategy == 'hybrid':
                f.write('**Híbrida (KNN + Moda)**\n\n')
                f.write('- KNN para variáveis numéricas\n')
                f.write('- Moda para variáveis categóricas\n')
                f.write('- Combina as vantagens de ambas as técnicas\n')
        
        logger.info(f'Relatório salvo em {report_path}')


def main():
    """Função principal para execução standalone"""
    from src.utils import load_config, load_data, save_data
    
    config = load_config()
    handler = MissingValuesHandler(config)
    
    # Carregar dados
    df = load_data(config['data']['raw_data_path'])
    
    # Aplicar tratamento
    df_imputed = handler.fit_transform(df)
    
    # Salvar resultado
    output_path = 'data/processed/tuberculosis_imputed.csv'
    save_data(df_imputed, output_path)
    
    print(f'\n✅ Tratamento de valores ausentes concluído!')
    print(f'📊 Dados salvos em: {output_path}')
    print(f'📈 Resultados em: results/preprocessing/missing_values/')


if __name__ == '__main__':
    main()
