"""
Módulo para balanceamento de classes (SMOTE, undersampling, etc)

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-25
Última Modificação: 2025-03-18

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
Balanceamento de Classes

Implementa SMOTE (Synthetic Minority Over-sampling Technique) conforme
descrito na Seção 4.2 da tese para balancear classes desbalanceadas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.utils import setup_logger

logger = setup_logger(__name__)


class ClassBalancer:
    """
    Classe para balanceamento de classes usando SMOTE.
    
    SMOTE (Synthetic Minority Over-sampling Technique) é a técnica
    recomendada na tese para lidar com desbalanceamento de classes.
    
    Gera amostras sintéticas da classe minoritária interpolando
    entre exemplos existentes.
    
    Attributes:
        config: Dicionário de configurações
        target_col: Nome da coluna alvo
        smote: Objeto SMOTE configurado
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o balanceador de classes.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        self.target_col = config['target']['column_name']
        self.random_state = config.get('random_state', 42)
        self.test_size = config.get('preprocessing', {}).get('test_size', 0.2)
        
        self.smote = SMOTE(random_state=self.random_state)
        
        self.output_dir = Path('results/preprocessing/class_balancing')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info('ClassBalancer inicializado com SMOTE')
    
    def analyze_class_distribution(
        self,
        y: pd.Series,
        label: str = 'Original'
    ) -> Dict[str, Any]:
        """
        Analisa a distribuição das classes.
        
        Args:
            y: Série com as classes
            label: Rótulo para identificação
        
        Returns:
            Dicionário com estatísticas de distribuição
        """
        logger.info(f'Analisando distribuição de classes ({label})')
        
        class_counts = y.value_counts().sort_index()
        class_percentages = y.value_counts(normalize=True).sort_index() * 100
        
        # Calcular razão de desbalanceamento
        if len(class_counts) >= 2:
            imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[1]
        else:
            imbalance_ratio = 1.0
        
        stats = {
            'counts': class_counts.to_dict(),
            'percentages': class_percentages.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'total': len(y)
        }
        
        logger.info(f'  Total: {len(y)}')
        for class_label, count in class_counts.items():
            pct = class_percentages[class_label]
            logger.info(f'  Classe {class_label}: {count} ({pct:.2f}%)')
        logger.info(f'  Razão de desbalanceamento: {imbalance_ratio:.2f}:1')
        
        return stats
    
    def plot_class_distribution(
        self,
        y: pd.Series,
        suffix: str = 'before'
    ) -> None:
        """
        Gera visualização da distribuição de classes.
        
        Args:
            y: Série com as classes
            suffix: Sufixo para nome do arquivo
        """
        logger.info(f'Gerando visualização de distribuição ({suffix})')
        
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=y, palette='viridis')
        plt.title(f'Distribuição das Classes ({suffix.title()})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Classe', fontsize=12)
        plt.ylabel('Contagem', fontsize=12)
        
        # Adicionar rótulos com contagem e percentual
        total = len(y)
        for p in ax.patches:
            height = p.get_height()
            percentage = (height / total) * 100
            ax.text(
                p.get_x() + p.get_width()/2.,
                height + total * 0.01,
                f'{int(height)}\n({percentage:.1f}%)',
                ha="center",
                fontsize=11,
                fontweight='bold'
            )
        
        plt.tight_layout()
        output_path = self.output_dir / f'class_distribution_{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'Visualização salva em {output_path}')
    
    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide dados em treino e teste.
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        logger.info(f'Dividindo dados em treino/teste ({1-self.test_size:.0%}/{self.test_size:.0%})')
        
        # Separar features e target
        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]
        
        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f'Treino: {X_train.shape[0]} amostras')
        logger.info(f'Teste: {X_test.shape[0]} amostras')
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica SMOTE para balancear classes no conjunto de treino.
        
        SMOTE gera amostras sintéticas da classe minoritária interpolando
        entre exemplos existentes no espaço de features.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        
        Returns:
            Tupla (X_train_balanced, y_train_balanced)
        """
        logger.info('Aplicando SMOTE')
        
        # Aplicar SMOTE
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        
        logger.info(f'Amostras antes: {X_train.shape[0]}')
        logger.info(f'Amostras depois: {X_train_balanced.shape[0]}')
        logger.info(f'Amostras sintéticas geradas: {X_train_balanced.shape[0] - X_train.shape[0]}')
        
        # Converter de volta para DataFrame/Series
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
        y_train_balanced = pd.Series(y_train_balanced, name=y_train.name)
        
        return X_train_balanced, y_train_balanced
    
    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide dados, aplica SMOTE e retorna conjuntos balanceados.
        
        Args:
            df: DataFrame com os dados
        
        Returns:
            Tupla (X_train_balanced, X_test, y_train_balanced, y_test)
        """
        logger.info('='*80)
        logger.info('INICIANDO BALANCEAMENTO DE CLASSES')
        logger.info('='*80)
        
        # Analisar distribuição original
        y_original = df[self.target_col]
        stats_original = self.analyze_class_distribution(y_original, 'Original')
        self.plot_class_distribution(y_original, suffix='before')
        
        # Dividir dados
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Analisar distribuição de treino antes do SMOTE
        stats_train_before = self.analyze_class_distribution(y_train, 'Treino Antes')
        
        # Aplicar SMOTE
        X_train_balanced, y_train_balanced = self.apply_smote(X_train, y_train)
        
        # Analisar distribuição de treino depois do SMOTE
        stats_train_after = self.analyze_class_distribution(y_train_balanced, 'Treino Depois')
        self.plot_class_distribution(y_train_balanced, suffix='after_smote')
        
        # Gerar relatório
        self._generate_report(
            stats_original,
            stats_train_before,
            stats_train_after,
            X_train.shape,
            X_train_balanced.shape,
            X_test.shape
        )
        
        logger.info('='*80)
        logger.info('BALANCEAMENTO DE CLASSES CONCLUÍDO')
        logger.info('='*80)
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def _generate_report(
        self,
        stats_original: Dict[str, Any],
        stats_train_before: Dict[str, Any],
        stats_train_after: Dict[str, Any],
        train_shape_before: Tuple[int, int],
        train_shape_after: Tuple[int, int],
        test_shape: Tuple[int, int]
    ) -> None:
        """
        Gera relatório do balanceamento de classes.
        
        Args:
            stats_original: Estatísticas da distribuição original
            stats_train_before: Estatísticas do treino antes do SMOTE
            stats_train_after: Estatísticas do treino depois do SMOTE
            train_shape_before: Dimensões do treino antes
            train_shape_after: Dimensões do treino depois
            test_shape: Dimensões do teste
        """
        logger.info('Gerando relatório de balanceamento de classes')
        
        report_path = self.output_dir / 'class_balancing_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('# Relatório de Balanceamento de Classes\n\n')
            
            f.write('## Técnica Utilizada: SMOTE\n\n')
            f.write('**SMOTE (Synthetic Minority Over-sampling Technique)**\n\n')
            f.write('- Gera amostras sintéticas da classe minoritária\n')
            f.write('- Interpola entre exemplos existentes no espaço de features\n')
            f.write('- Evita overfitting causado por simples duplicação\n')
            f.write('- Técnica recomendada pela tese para dados de tuberculose\n\n')
            
            f.write('## Distribuição Original\n\n')
            f.write(f'- **Total:** {stats_original["total"]:,} amostras\n')
            f.write(f'- **Razão de desbalanceamento:** {stats_original["imbalance_ratio"]:.2f}:1\n\n')
            f.write('| Classe | Contagem | Percentual (%) |\n')
            f.write('|--------|----------|----------------|\n')
            for class_label in sorted(stats_original['counts'].keys()):
                count = stats_original['counts'][class_label]
                pct = stats_original['percentages'][class_label]
                f.write(f'| {class_label} | {count:,} | {pct:.2f} |\n')
            f.write('\n')
            
            f.write('## Divisão Treino/Teste\n\n')
            f.write(f'- **Treino:** {train_shape_before[0]:,} amostras ({(1-self.test_size)*100:.0f}%)\n')
            f.write(f'- **Teste:** {test_shape[0]:,} amostras ({self.test_size*100:.0f}%)\n\n')
            
            f.write('## Aplicação do SMOTE (Conjunto de Treino)\n\n')
            f.write('### Antes do SMOTE\n\n')
            f.write(f'- **Total:** {stats_train_before["total"]:,} amostras\n')
            f.write(f'- **Razão:** {stats_train_before["imbalance_ratio"]:.2f}:1\n\n')
            
            f.write('### Depois do SMOTE\n\n')
            f.write(f'- **Total:** {stats_train_after["total"]:,} amostras\n')
            f.write(f'- **Razão:** {stats_train_after["imbalance_ratio"]:.2f}:1\n')
            f.write(f'- **Amostras sintéticas geradas:** {train_shape_after[0] - train_shape_before[0]:,}\n\n')
            
            f.write('| Classe | Antes | Depois | Diferença |\n')
            f.write('|--------|-------|--------|----------|\n')
            for class_label in sorted(stats_train_before['counts'].keys()):
                before = stats_train_before['counts'][class_label]
                after = stats_train_after['counts'][class_label]
                diff = after - before
                f.write(f'| {class_label} | {before:,} | {after:,} | +{diff:,} |\n')
            f.write('\n')
            
            f.write('## Observações Importantes\n\n')
            f.write('1. **SMOTE aplicado apenas no conjunto de treino** para evitar data leakage\n')
            f.write('2. **Conjunto de teste mantido intacto** para avaliação realista\n')
            f.write('3. **Classes perfeitamente balanceadas** no treino após SMOTE\n')
            f.write('4. **Amostras sintéticas** são interpolações, não duplicatas\n')
        
        logger.info(f'Relatório salvo em {report_path}')


def main():
    """Função principal para execução standalone"""
    from src.utils import load_config, load_data, save_data
    
    config = load_config()
    balancer = ClassBalancer(config)
    
    # Carregar dados
    df = load_data('data/processed/tuberculosis_outliers_treated.csv')
    
    # Aplicar balanceamento
    X_train, X_test, y_train, y_test = balancer.fit_transform(df)
    
    # Salvar conjuntos
    train_df = X_train.copy()
    train_df[config['target']['column_name']] = y_train
    save_data(train_df, 'data/processed/train_balanced.csv')
    
    test_df = X_test.copy()
    test_df[config['target']['column_name']] = y_test
    save_data(test_df, 'data/processed/test.csv')
    
    print(f'\n✅ Balanceamento de classes concluído!')
    print(f'📊 Treino balanceado: data/processed/train_balanced.csv')
    print(f'📊 Teste: data/processed/test.csv')
    print(f'📈 Resultados em: results/preprocessing/class_balancing/')


if __name__ == '__main__':
    main()
