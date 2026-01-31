"""
Módulo para detecção e tratamento de outliers

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-18
Última Modificação: 2025-03-12

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
Tratamento de Outliers

Implementa detecção e tratamento de outliers usando Isolation Forest
conforme descrito na Seção 4.2 da tese.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging

from sklearn.ensemble import IsolationForest

from src.utils import setup_logger

logger = setup_logger(__name__)


class OutliersHandler:
    """
    Classe para detecção e tratamento de outliers.
    
    Implementa Isolation Forest conforme descrito na tese.
    Isolation Forest é eficaz para detectar outliers em datasets multivariados.
    
    Attributes:
        config: Dicionário de configurações
        contamination: Proporção esperada de outliers
        detector: Modelo Isolation Forest
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o handler de outliers.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        self.contamination = config.get('preprocessing', {}).get('outliers_contamination', 0.1)
        self.target_col = config['target']['column_name']
        self.random_state = config.get('random_state', 42)
        
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        self.output_dir = Path('results/preprocessing/outliers')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'OutliersHandler inicializado com contamination={self.contamination}')
    
    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        col: str,
        factor: float = 1.5
    ) -> Tuple[pd.Series, float, float]:
        """
        Detecta outliers usando método IQR (Interquartile Range).
        
        Args:
            df: DataFrame com os dados
            col: Nome da coluna
            factor: Fator multiplicador do IQR (padrão: 1.5)
        
        Returns:
            Tupla (outliers, lower_bound, upper_bound)
        """
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        return outliers, lower_bound, upper_bound
    
    def analyze_outliers_iqr(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analisa outliers em todas as variáveis numéricas usando IQR.
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            Dicionário com estatísticas de outliers por variável
        """
        logger.info('Analisando outliers usando método IQR')
        
        # Separar features e target
        if self.target_col in df.columns:
            X = df.drop(self.target_col, axis=1)
        else:
            X = df
        
        # Identificar variáveis numéricas
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        logger.info(f'Total de variáveis numéricas: {len(numeric_cols)}')
        
        outliers_summary = {}
        
        for col in numeric_cols:
            outliers, lower_bound, upper_bound = self.detect_outliers_iqr(X, col)
            outliers_pct = len(outliers) / len(X) * 100
            
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': outliers_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outliers_pct > 0:
                logger.info(f'  {col}: {len(outliers)} outliers ({outliers_pct:.2f}%)')
        
        return outliers_summary
    
    def detect_outliers_isolation_forest(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detecta outliers usando Isolation Forest.
        
        Conforme descrito na tese, Isolation Forest é eficaz para
        detectar outliers em datasets multivariados de alta dimensão.
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            Array booleano indicando outliers (True = outlier)
        """
        logger.info('Detectando outliers usando Isolation Forest')
        
        # Separar features e target
        if self.target_col in df.columns:
            X = df.drop(self.target_col, axis=1)
        else:
            X = df
        
        # Selecionar apenas variáveis numéricas
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X_numeric = X[numeric_cols]
        
        # Treinar Isolation Forest
        predictions = self.detector.fit_predict(X_numeric)
        
        # -1 indica outlier, 1 indica inlier
        outliers_mask = predictions == -1
        
        n_outliers = outliers_mask.sum()
        outliers_pct = (n_outliers / len(df)) * 100
        
        logger.info(f'Outliers detectados: {n_outliers} ({outliers_pct:.2f}%)')
        
        return outliers_mask
    
    def plot_outliers_distribution(
        self,
        df: pd.DataFrame,
        outliers_summary: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Gera visualizações da distribuição de outliers.
        
        Args:
            df: DataFrame com os dados
            outliers_summary: Sumário de outliers por variável
        """
        logger.info('Gerando visualizações de outliers')
        
        # Separar features
        if self.target_col in df.columns:
            X = df.drop(self.target_col, axis=1)
        else:
            X = df
        
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Gráfico de barras com percentual de outliers por variável
        if outliers_summary:
            variables = list(outliers_summary.keys())
            percentages = [outliers_summary[var]['percentage'] for var in variables]
            
            plt.figure(figsize=(12, 8))
            plt.barh(variables, percentages, color='coral')
            plt.xlabel('Percentual de Outliers (%)', fontsize=12)
            plt.ylabel('Variável', fontsize=12)
            plt.title('Percentual de Outliers por Variável (Método IQR)', 
                     fontsize=14, fontweight='bold')
            plt.axvline(x=self.contamination * 100, color='red', linestyle='--', 
                       linewidth=2, label=f'Limiar: {self.contamination*100:.1f}%')
            plt.legend()
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            output_path = self.output_dir / 'outliers_percentage_by_variable.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f'Gráfico de percentuais salvo em {output_path}')
        
        # Boxplots para variáveis com mais outliers (top 10)
        if outliers_summary:
            top_vars = sorted(outliers_summary.items(), 
                            key=lambda x: x[1]['percentage'], 
                            reverse=True)[:10]
            
            if top_vars:
                fig, axes = plt.subplots(5, 2, figsize=(14, 16))
                axes = axes.flatten()
                
                for idx, (var, stats) in enumerate(top_vars):
                    if var in X.columns:
                        ax = axes[idx]
                        ax.boxplot(X[var].dropna())
                        ax.set_title(f'{var}\n({stats["percentage"]:.2f}% outliers)', 
                                   fontsize=10, fontweight='bold')
                        ax.set_ylabel('Valor')
                        ax.grid(True, alpha=0.3)
                
                # Remover subplots vazios
                for idx in range(len(top_vars), len(axes)):
                    fig.delaxes(axes[idx])
                
                plt.tight_layout()
                output_path = self.output_dir / 'outliers_boxplots_top10.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f'Boxplots salvos em {output_path}')
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        method: str = 'isolation_forest'
    ) -> pd.DataFrame:
        """
        Detecta e remove outliers.
        
        Args:
            df: DataFrame com os dados
            method: Método de detecção ('isolation_forest' ou 'iqr')
        
        Returns:
            DataFrame sem outliers
        """
        logger.info('='*80)
        logger.info('INICIANDO TRATAMENTO DE OUTLIERS')
        logger.info('='*80)
        
        # Analisar outliers usando IQR (para visualização)
        outliers_summary = self.analyze_outliers_iqr(df)
        self.plot_outliers_distribution(df, outliers_summary)
        
        # Detectar outliers usando método selecionado
        if method == 'isolation_forest':
            outliers_mask = self.detect_outliers_isolation_forest(df)
        elif method == 'iqr':
            # Implementar remoção baseada em IQR se necessário
            logger.warning('Método IQR não implementado para remoção. Usando Isolation Forest.')
            outliers_mask = self.detect_outliers_isolation_forest(df)
        else:
            raise ValueError(f'Método inválido: {method}')
        
        # Remover outliers
        df_clean = df[~outliers_mask].copy()
        
        n_removed = outliers_mask.sum()
        pct_removed = (n_removed / len(df)) * 100
        
        logger.info(f'Dimensões antes: {df.shape}')
        logger.info(f'Dimensões depois: {df_clean.shape}')
        logger.info(f'Registros removidos: {n_removed} ({pct_removed:.2f}%)')
        
        # Gerar relatório
        self._generate_report(outliers_summary, n_removed, pct_removed, method)
        
        logger.info('='*80)
        logger.info('TRATAMENTO DE OUTLIERS CONCLUÍDO')
        logger.info('='*80)
        
        return df_clean
    
    def _generate_report(
        self,
        outliers_summary: Dict[str, Dict[str, Any]],
        n_removed: int,
        pct_removed: float,
        method: str
    ) -> None:
        """
        Gera relatório do tratamento de outliers.
        
        Args:
            outliers_summary: Sumário de outliers por variável
            n_removed: Número de registros removidos
            pct_removed: Percentual de registros removidos
            method: Método utilizado
        """
        logger.info('Gerando relatório de tratamento de outliers')
        
        report_path = self.output_dir / 'outliers_treatment_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('# Relatório de Tratamento de Outliers\n\n')
            
            f.write(f'## Configuração\n\n')
            f.write(f'- **Método:** {method.upper()}\n')
            f.write(f'- **Contamination (Isolation Forest):** {self.contamination}\n\n')
            
            f.write('## Análise de Outliers (Método IQR)\n\n')
            if outliers_summary:
                f.write('| Variável | Outliers | Percentual (%) |\n')
                f.write('|----------|----------|----------------|\n')
                for var, stats in sorted(outliers_summary.items(), 
                                        key=lambda x: x[1]['percentage'], 
                                        reverse=True)[:20]:
                    f.write(f'| {var} | {stats["count"]} | {stats["percentage"]:.2f} |\n')
                f.write('\n')
            
            f.write('## Ação Tomada\n\n')
            f.write(f'- **Registros removidos:** {n_removed} ({pct_removed:.2f}%)\n')
            f.write(f'- **Método utilizado:** {method.upper()}\n\n')
            
            f.write('## Sobre o Isolation Forest\n\n')
            f.write('O Isolation Forest é um algoritmo de detecção de anomalias que:\n')
            f.write('- Isola observações construindo árvores de decisão aleatórias\n')
            f.write('- Outliers são isolados mais rapidamente (menor profundidade)\n')
            f.write('- Eficaz para datasets multivariados de alta dimensão\n')
            f.write('- Não assume distribuição específica dos dados\n')
        
        logger.info(f'Relatório salvo em {report_path}')


def main():
    """Função principal para execução standalone"""
    from src.utils import load_config, load_data, save_data
    
    config = load_config()
    handler = OutliersHandler(config)
    
    # Carregar dados
    df = load_data('data/processed/tuberculosis_correlation_treated.csv')
    
    # Aplicar tratamento
    df_clean = handler.fit_transform(df, method='isolation_forest')
    
    # Salvar resultado
    save_data(df_clean, 'data/processed/tuberculosis_outliers_treated.csv')
    
    print(f'\n✅ Tratamento de outliers concluído!')
    print(f'📊 Dados salvos em: data/processed/tuberculosis_outliers_treated.csv')
    print(f'📈 Resultados em: results/preprocessing/outliers/')


if __name__ == '__main__':
    main()
