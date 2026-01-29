"""
Módulo para análise e tratamento de correlações entre variáveis

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-20
Última Modificação: 2025-03-14

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
Tratamento de Correlação

Identifica e trata variáveis altamente correlacionadas conforme descrito na Seção 4.2 da tese.
Remove variáveis redundantes mantendo aquelas com maior correlação com a variável alvo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import logging
import networkx as nx

from src.utils import setup_logger

logger = setup_logger(__name__)


class CorrelationHandler:
    """
    Classe para tratamento de variáveis altamente correlacionadas.
    
    Implementa a estratégia descrita na tese:
    1. Identifica pares de variáveis com |correlação| > threshold
    2. Para cada par, mantém a variável com maior correlação com o alvo
    3. Remove variáveis redundantes
    
    Attributes:
        config: Dicionário de configurações
        threshold: Limiar de correlação (padrão: 0.8)
        variables_to_remove: Lista de variáveis a serem removidas
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o handler de correlação.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        self.threshold = config.get('preprocessing', {}).get('correlation_threshold', 0.8)
        self.target_col = config['target']['column_name']
        self.variables_to_remove = []
        self.output_dir = Path('results/preprocessing/correlation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'CorrelationHandler inicializado com threshold={self.threshold}')
    
    def analyze_correlation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Analisa correlações entre variáveis.
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            Tupla (matriz_correlacao, pares_alta_correlacao)
        """
        logger.info('Analisando correlações')
        
        # Separar features e target
        if self.target_col in df.columns:
            X = df.drop(self.target_col, axis=1)
        else:
            X = df
        
        # Calcular matriz de correlação
        correlation_matrix = X.corr()
        
        # Identificar pares altamente correlacionados
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > self.threshold:
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((var1, var2, corr_value))
        
        # Ordenar por valor absoluto de correlação
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        logger.info(f'Total de pares com |correlação| > {self.threshold}: {len(high_corr_pairs)}')
        
        return correlation_matrix, high_corr_pairs
    
    def plot_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame,
        suffix: str = 'before'
    ) -> None:
        """
        Gera visualização da matriz de correlação.
        
        Args:
            correlation_matrix: Matriz de correlação
            suffix: Sufixo para nome do arquivo
        """
        logger.info(f'Gerando visualização da matriz de correlação ({suffix})')
        
        # Matriz de correlação completa (triângulo inferior)
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap='coolwarm',
            annot=False,
            center=0,
            vmin=-1,
            vmax=1,
            square=True
        )
        plt.title(f'Matriz de Correlação ({suffix.title()})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f'correlation_matrix_{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'Matriz de correlação salva em {output_path}')
        
        # Distribuição das correlações
        plt.figure(figsize=(10, 6))
        corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
        sns.histplot(corr_values, bins=50, kde=True, color='steelblue')
        plt.axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Limiar: {self.threshold}')
        plt.axvline(x=-self.threshold, color='red', linestyle='--', linewidth=2)
        plt.title(f'Distribuição dos Coeficientes de Correlação ({suffix.title()})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Coeficiente de Correlação', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        output_path = self.output_dir / f'correlation_distribution_{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'Distribuição de correlações salva em {output_path}')
    
    def identify_groups(self, high_corr_pairs: List[Tuple[str, str, float]]) -> List[Set[str]]:
        """
        Identifica grupos de variáveis correlacionadas usando teoria dos grafos.
        
        Args:
            high_corr_pairs: Lista de pares altamente correlacionados
        
        Returns:
            Lista de conjuntos de variáveis correlacionadas
        """
        logger.info('Identificando grupos de variáveis correlacionadas')
        
        # Criar grafo não direcionado
        G = nx.Graph()
        for var1, var2, corr in high_corr_pairs:
            G.add_edge(var1, var2, weight=abs(corr))
        
        # Encontrar componentes conectados
        connected_components = list(nx.connected_components(G))
        
        logger.info(f'Total de grupos identificados: {len(connected_components)}')
        for i, component in enumerate(connected_components, 1):
            logger.info(f'  Grupo {i}: {len(component)} variáveis')
        
        return connected_components
    
    def select_variables_to_keep(
        self,
        df: pd.DataFrame,
        groups: List[Set[str]]
    ) -> List[str]:
        """
        Seleciona quais variáveis manter em cada grupo.
        
        Estratégia: Mantém a variável com maior correlação absoluta com o alvo.
        
        Args:
            df: DataFrame com os dados
            groups: Lista de grupos de variáveis correlacionadas
        
        Returns:
            Lista de variáveis a remover
        """
        logger.info('Selecionando variáveis a manter/remover')
        
        variables_to_remove = []
        
        if self.target_col not in df.columns:
            logger.warning('Variável alvo não encontrada. Usando primeira variável de cada grupo.')
            for group in groups:
                group_list = list(group)
                # Manter primeira variável, remover as demais
                variables_to_remove.extend(group_list[1:])
            return variables_to_remove
        
        # Calcular correlação de cada variável com o alvo
        target_correlations = df.corr()[self.target_col].abs()
        
        for group in groups:
            group_list = list(group)
            
            # Encontrar variável com maior correlação com o alvo
            group_corrs = {var: target_correlations.get(var, 0) for var in group_list}
            var_to_keep = max(group_corrs, key=group_corrs.get)
            
            # Remover as demais
            vars_to_remove = [var for var in group_list if var != var_to_keep]
            variables_to_remove.extend(vars_to_remove)
            
            logger.info(f'  Grupo: mantendo {var_to_keep} (corr={group_corrs[var_to_keep]:.4f}), '
                       f'removendo {len(vars_to_remove)} variáveis')
        
        logger.info(f'Total de variáveis a remover: {len(variables_to_remove)}')
        
        return variables_to_remove
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa e remove variáveis altamente correlacionadas.
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            DataFrame com variáveis correlacionadas removidas
        """
        logger.info('='*80)
        logger.info('INICIANDO TRATAMENTO DE CORRELAÇÃO')
        logger.info('='*80)
        
        # Analisar correlações antes
        corr_matrix_before, high_corr_pairs = self.analyze_correlation(df)
        self.plot_correlation_matrix(corr_matrix_before, suffix='before')
        
        if not high_corr_pairs:
            logger.info('Nenhum par de variáveis altamente correlacionadas encontrado')
            logger.info('='*80)
            logger.info('TRATAMENTO DE CORRELAÇÃO CONCLUÍDO (SEM ALTERAÇÕES)')
            logger.info('='*80)
            return df
        
        # Identificar grupos
        groups = self.identify_groups(high_corr_pairs)
        
        # Selecionar variáveis a remover
        self.variables_to_remove = self.select_variables_to_keep(df, groups)
        
        # Remover variáveis
        df_reduced = df.drop(columns=self.variables_to_remove)
        
        logger.info(f'Dimensões antes: {df.shape}')
        logger.info(f'Dimensões depois: {df_reduced.shape}')
        logger.info(f'Variáveis removidas: {len(self.variables_to_remove)}')
        
        # Analisar correlações depois
        corr_matrix_after, high_corr_pairs_after = self.analyze_correlation(df_reduced)
        self.plot_correlation_matrix(corr_matrix_after, suffix='after')
        
        # Gerar relatório
        self._generate_report(high_corr_pairs, groups, self.variables_to_remove, 
                            len(high_corr_pairs_after))
        
        logger.info('='*80)
        logger.info('TRATAMENTO DE CORRELAÇÃO CONCLUÍDO')
        logger.info('='*80)
        
        return df_reduced
    
    def _generate_report(
        self,
        high_corr_pairs: List[Tuple[str, str, float]],
        groups: List[Set[str]],
        variables_removed: List[str],
        remaining_high_corr: int
    ) -> None:
        """
        Gera relatório do tratamento de correlação.
        
        Args:
            high_corr_pairs: Pares altamente correlacionados antes
            groups: Grupos identificados
            variables_removed: Variáveis removidas
            remaining_high_corr: Pares altamente correlacionados restantes
        """
        logger.info('Gerando relatório de tratamento de correlação')
        
        report_path = self.output_dir / 'correlation_treatment_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('# Relatório de Tratamento de Correlação\n\n')
            
            f.write(f'## Configuração\n\n')
            f.write(f'- **Limiar de correlação:** {self.threshold}\n')
            f.write(f'- **Variável alvo:** {self.target_col}\n\n')
            
            f.write('## Análise Inicial\n\n')
            f.write(f'- **Pares altamente correlacionados:** {len(high_corr_pairs)}\n')
            f.write(f'- **Grupos identificados:** {len(groups)}\n\n')
            
            if high_corr_pairs:
                f.write('### Top 10 Pares Mais Correlacionados\n\n')
                f.write('| Variável 1 | Variável 2 | Correlação |\n')
                f.write('|------------|------------|------------|\n')
                for var1, var2, corr in high_corr_pairs[:10]:
                    f.write(f'| {var1} | {var2} | {corr:.4f} |\n')
                f.write('\n')
            
            f.write('## Ação Tomada\n\n')
            f.write(f'- **Variáveis removidas:** {len(variables_removed)}\n')
            f.write(f'- **Pares altamente correlacionados restantes:** {remaining_high_corr}\n\n')
            
            if variables_removed:
                f.write('### Variáveis Removidas\n\n')
                for var in variables_removed:
                    f.write(f'- {var}\n')
                f.write('\n')
            
            f.write('## Estratégia Aplicada\n\n')
            f.write('Para cada grupo de variáveis correlacionadas:\n')
            f.write('1. Calcular correlação de cada variável com a variável alvo\n')
            f.write('2. Manter a variável com maior correlação absoluta com o alvo\n')
            f.write('3. Remover as demais variáveis do grupo\n\n')
            
            f.write('Esta estratégia garante que mantemos as variáveis mais informativas '
                   'para predição do desfecho.\n')
        
        logger.info(f'Relatório salvo em {report_path}')


def main():
    """Função principal para execução standalone"""
    from src.utils import load_config, load_data, save_data
    
    config = load_config()
    handler = CorrelationHandler(config)
    
    # Carregar dados
    df = load_data('data/processed/tuberculosis_encoded.csv')
    
    # Aplicar tratamento
    df_reduced = handler.fit_transform(df)
    
    # Salvar resultado
    save_data(df_reduced, 'data/processed/tuberculosis_correlation_treated.csv')
    
    print(f'\n✅ Tratamento de correlação concluído!')
    print(f'📊 Dados salvos em: data/processed/tuberculosis_correlation_treated.csv')
    print(f'📈 Resultados em: results/preprocessing/correlation/')


if __name__ == '__main__':
    main()
