"""
Módulo para Balanceamento de Classes

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-02-25
Última Modificação: 2025-11-30

Descrição:
    Implementa SMOTE (Synthetic Minority Over-sampling Technique) conforme
    descrito na Seção 4.2 da tese para balancear classes desbalanceadas.
    
    IMPORTANTE: SMOTE deve ser aplicado DEPOIS de:
    1. Tratamento de valores ausentes
    2. Encoding de variáveis categóricas
    3. Normalização/Padronização

Licença: MIT
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
    
    IMPORTANTE: SMOTE SÓ FUNCIONA COM DADOS NUMÉRICOS!
    Deve ser aplicado DEPOIS de encoding de variáveis categóricas.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o balanceador de classes.
        
        Parâmetros:
        -----------
        config : Dict[str, Any]
            Dicionário de configurações
        """
        self.config = config
        self.target_col = config['target']['column_name']
        self.random_state = config.get('random_state', 42)
        self.test_size = config.get('preprocessing', {}).get('test_size', 0.2)
        
        self.smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        
        self.output_dir = Path('results/preprocessing/class_balancing')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info('✅ ClassBalancer inicializado com SMOTE')
    
    def analyze_class_distribution(
        self,
        y: pd.Series,
        label: str = 'Original'
    ) -> Dict[str, Any]:
        """
        Analisa a distribuição das classes.
        
        Parâmetros:
        -----------
        y : pd.Series
            Série com as classes
        label : str
            Rótulo para identificação
        
        Retorna:
        --------
        Dict[str, Any]
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
            logger.info(f'    Classe {class_label}: {count:,} ({pct:.2f}%)')
        logger.info(f'  Razão de desbalanceamento: {imbalance_ratio:.2f}:1')
        
        return stats
    
    def plot_class_distribution(
        self,
        y: pd.Series,
        suffix: str = 'before'
    ) -> None:
        """
        Gera visualização da distribuição de classes.
        
        Parâmetros:
        -----------
        y : pd.Series
            Série com as classes
        suffix : str
            Sufixo para nome do arquivo
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
                f'{int(height):,}\n({percentage:.1f}%)',
                ha="center",
                fontsize=11,
                fontweight='bold'
            )
        
        plt.tight_layout()
        output_path = self.output_dir / f'class_distribution_{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'  Visualização salva em {output_path}')
    
    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide dados em treino e teste.
        
        IMPORTANTE: Split é feito ANTES de SMOTE para evitar data leakage!
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados (já com encoding aplicado)
        
        Retorna:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            (X_train, X_test, y_train, y_test)
        """
        logger.info(f'Dividindo dados em treino/teste ({1-self.test_size:.0%}/{self.test_size:.0%})')
        
        # Separar features e target
        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]
        
        # Split estratificado (mantém proporção de classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f'  Treino: {X_train.shape[0]:,} amostras')
        logger.info(f'  Teste: {X_test.shape[0]:,} amostras')
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica SMOTE para balancear classes no conjunto de treino.
        
        IMPORTANTE: SMOTE SÓ FUNCIONA COM DADOS NUMÉRICOS!
        Deve ser aplicado DEPOIS de encoding.
        
        SMOTE gera amostras sintéticas da classe minoritária interpolando
        entre exemplos existentes no espaço de features.
        
        Parâmetros:
        -----------
        X_train : pd.DataFrame
            Features de treino (DEVE SER NUMÉRICA!)
        y_train : pd.Series
            Target de treino
        
        Retorna:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (X_train_balanced, y_train_balanced)
        """
        logger.info('Aplicando SMOTE...')
        
        # Validar que X_train é numérico
        if X_train.select_dtypes(include=['object', 'category']).shape[1] > 0:
            raise ValueError(
                "❌ ERRO: X_train contém variáveis categóricas! "
                "SMOTE só funciona com dados numéricos. "
                "Aplique encoding ANTES de SMOTE!"
            )
        
        # Aplicar SMOTE
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        
        logger.info(f'  Amostras antes: {X_train.shape[0]:,}')
        logger.info(f'  Amostras depois: {X_train_balanced.shape[0]:,}')
        logger.info(f'  Amostras sintéticas geradas: {X_train_balanced.shape[0] - X_train.shape[0]:,}')
        
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
        
        PIPELINE CORRETO:
        1. Carregamento de dados
        2. Tratamento de valores ausentes
        3. Encoding de variáveis categóricas
        4. Normalização/Padronização
        5. Split treino/teste
        6. SMOTE (APENAS no treino!)
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com os dados (já processado!)
        
        Retorna:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            (X_train_balanced, X_test, y_train_balanced, y_test)
        """
        logger.info('='*80)
        logger.info('INICIANDO BALANCEAMENTO DE CLASSES COM SMOTE')
        logger.info('='*80)
        
        # Analisar distribuição original
        y_original = df[self.target_col]
        stats_original = self.analyze_class_distribution(y_original, 'Original')
        self.plot_class_distribution(y_original, suffix='before')
        
        # Dividir dados (ANTES de SMOTE para evitar data leakage)
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Analisar distribuição de treino antes do SMOTE
        stats_train_before = self.analyze_class_distribution(y_train, 'Treino Antes')
        
        # Aplicar SMOTE (APENAS no treino)
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
        
        Parâmetros:
        -----------
        stats_original : Dict[str, Any]
            Estatísticas da distribuição original
        stats_train_before : Dict[str, Any]
            Estatísticas do treino antes do SMOTE
        stats_train_after : Dict[str, Any]
            Estatísticas do treino depois do SMOTE
        train_shape_before : Tuple[int, int]
            Dimensões do treino antes
        train_shape_after : Tuple[int, int]
            Dimensões do treino depois
        test_shape : Tuple[int, int]
            Dimensões do teste
        """
        logger.info('Gerando relatório de balanceamento de classes')
        
        logger.info('\n📊 RESUMO DO BALANCEAMENTO:')
        logger.info(f'  Dataset Original: {stats_original["total"]:,} amostras')
        logger.info(f'  Treino Antes SMOTE: {train_shape_before[0]:,} amostras')
        logger.info(f'  Treino Depois SMOTE: {train_shape_after[0]:,} amostras')
        logger.info(f'  Teste (sem SMOTE): {test_shape[0]:,} amostras')
        
        logger.info(f'\n  Razão de desbalanceamento:')
        logger.info(f'    Antes: {stats_train_before["imbalance_ratio"]:.2f}:1')
        logger.info(f'    Depois: {stats_train_after["imbalance_ratio"]:.2f}:1')


if __name__ == "__main__":
    logger.info("Class Balancing Module - Exemplo de uso")
