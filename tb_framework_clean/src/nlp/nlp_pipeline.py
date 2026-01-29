"""
Pipeline completo de Natural Language Processing

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-07-20
Última Modificação: 2025-09-15

Descrição:
    Este módulo faz parte do framework multi-paradigma desenvolvido para predição
    de abandono de tratamento em pacientes com tuberculose. Implementa o pipeline
    completo de NLP, desde a geração de narrativas até a extração de features
    para integração com o ensemble multi-paradigma.


"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import json

from src.nlp.synthetic_narratives import SyntheticNarrativeGenerator
from src.nlp.text_feature_extraction import TextFeatureExtractor

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Pipeline completo de processamento de linguagem natural.
    
    Integra:
    1. Geração de narrativas clínicas sintéticas
    2. Extração de features de texto (TF-IDF, LDA, estatísticas)
    3. Preparação de features para o ensemble
    
    Attributes:
        narrative_generator: Gerador de narrativas sintéticas
        feature_extractor: Extrator de features de texto
        config: Configurações do pipeline
        output_dir: Diretório para resultados
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o pipeline NLP.
        
        Args:
            config: Dicionário de configurações
        """
        self.config = config
        self.output_dir = Path('results/nlp')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        nlp_config = config.get('nlp', {})
        
        self.narrative_generator = SyntheticNarrativeGenerator(
            output_dir=str(self.output_dir / 'narratives')
        )
        
        self.feature_extractor = TextFeatureExtractor(
            max_features=nlp_config.get('max_features', 100),
            n_topics=nlp_config.get('n_topics', 10),
            output_dir=str(self.output_dir / 'features')
        )
        
        logger.info('NLPPipeline inicializado')
    
    def generate_narratives(
        self,
        df: pd.DataFrame,
        n_narratives: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Gera narrativas clínicas sintéticas.
        
        Args:
            df: DataFrame com dados dos pacientes
            n_narratives: Número de narrativas a gerar (None = todas)
        
        Returns:
            DataFrame com coluna 'narrativa_clinica' adicionada
        """
        logger.info('Etapa 1: Gerando narrativas clínicas sintéticas')
        
        df_with_narratives = self.narrative_generator.generate_narratives(
            df, n_narratives=n_narratives
        )
        
        # Estatísticas
        stats = self.narrative_generator.get_statistics(df_with_narratives)
        logger.info(f'Narrativas geradas: {stats}')
        
        return df_with_narratives
    
    def extract_features(
        self,
        df: pd.DataFrame,
        text_column: str = 'narrativa_clinica',
        fit: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extrai features de texto das narrativas.
        
        Args:
            df: DataFrame com narrativas
            text_column: Nome da coluna com textos
            fit: Se True, treina os modelos; se False, apenas transforma
        
        Returns:
            Tupla (features_numéricas, features_estatísticas)
        """
        logger.info('Etapa 2: Extraindo features de texto')
        
        # Verificar se coluna existe
        if text_column not in df.columns:
            logger.error(f'Coluna {text_column} não encontrada no DataFrame')
            raise ValueError(f'Coluna {text_column} não encontrada')
        
        # Extrair textos
        texts = df[text_column].fillna('').tolist()
        
        # Extrair features
        numeric_features, stats_features = self.feature_extractor.extract_all_features(
            texts, fit=fit
        )
        
        logger.info(f'Features extraídas: {numeric_features.shape[1]} numéricas, {stats_features.shape[1]} estatísticas')
        
        # Salvar modelos se foi fit
        if fit:
            self.feature_extractor.save_models()
        
        return numeric_features, stats_features
    
    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        n_narratives: Optional[int] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Executa o pipeline completo de NLP.
        
        Args:
            df: DataFrame com dados dos pacientes
            n_narratives: Número de narrativas a gerar
            fit: Se True, treina os modelos
        
        Returns:
            Tupla (df_com_narrativas, features_numéricas, features_estatísticas)
        """
        logger.info('Executando pipeline completo de NLP')
        
        # 1. Gerar narrativas
        df_with_narratives = self.generate_narratives(df, n_narratives)
        
        # 2. Extrair features
        numeric_features, stats_features = self.extract_features(
            df_with_narratives,
            fit=fit
        )
        
        # 3. Salvar resultados
        self._save_results(df_with_narratives, numeric_features, stats_features)
        
        logger.info('Pipeline NLP concluído')
        
        return df_with_narratives, numeric_features, stats_features
    
    def _save_results(
        self,
        df: pd.DataFrame,
        numeric_features: np.ndarray,
        stats_features: pd.DataFrame
    ) -> None:
        """
        Salva resultados do pipeline.
        
        Args:
            df: DataFrame com narrativas
            numeric_features: Features numéricas
            stats_features: Features estatísticas
        """
        logger.info('Salvando resultados do pipeline NLP')
        
        # Salvar narrativas
        narratives_file = self.output_dir / 'narratives_complete.csv'
        df[['narrativa_clinica']].to_csv(narratives_file, index=False)
        
        # Salvar features numéricas
        np.save(
            self.output_dir / 'numeric_features.npy',
            numeric_features
        )
        
        # Salvar features estatísticas
        stats_file = self.output_dir / 'text_statistics.csv'
        stats_features.to_csv(stats_file, index=False)
        
        logger.info(f'Resultados salvos em {self.output_dir}')
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Retorna informações sobre as features mais importantes.
        
        Returns:
            Dicionário com termos e tópicos importantes
        """
        logger.info('Obtendo features mais importantes')
        
        importance = {
            'top_terms': self.feature_extractor.get_top_terms(n_terms=20),
            'topic_words': self.feature_extractor.get_topic_words(n_words=10)
        }
        
        # Salvar
        importance_file = self.output_dir / 'feature_importance.json'
        with open(importance_file, 'w', encoding='utf-8') as f:
            json.dump(importance, f, ensure_ascii=False, indent=2)
        
        logger.info(f'Importância de features salva em {importance_file}')
        
        return importance
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Gera relatório do pipeline NLP.
        
        Args:
            df: DataFrame processado
        
        Returns:
            Caminho do arquivo de relatório
        """
        logger.info('Gerando relatório do pipeline NLP')
        
        report = []
        report.append('# Relatório do Pipeline NLP\n')
        report.append('## Resumo Executivo\n')
        
        # Estatísticas de narrativas
        if 'narrativa_clinica' in df.columns:
            narratives = df['narrativa_clinica'].dropna()
            report.append(f'- Total de narrativas geradas: {len(narratives)}')
            report.append(f'- Comprimento médio: {narratives.str.len().mean():.1f} caracteres')
            report.append(f'- Média de palavras: {narratives.str.split().str.len().mean():.1f}')
        
        report.append('\n## Features Extraídas\n')
        
        # Termos importantes
        top_terms = self.feature_extractor.get_top_terms(n_terms=15)
        if top_terms:
            report.append('### Termos Mais Importantes (TF-IDF)\n')
            for i, term in enumerate(top_terms, 1):
                report.append(f'{i}. {term}')
        
        report.append('\n### Tópicos Identificados (LDA)\n')
        
        # Tópicos
        topics = self.feature_extractor.get_topic_words(n_words=5)
        for topic_id, words in topics.items():
            report.append(f'\n**Tópico {topic_id + 1}:** {", ".join(words)}')
        
        # Salvar relatório
        report_text = '\n'.join(report)
        report_file = self.output_dir / 'nlp_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f'Relatório salvo em {report_file}')
        
        return str(report_file)
