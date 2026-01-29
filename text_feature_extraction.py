"""
Módulo para extração de features de textos clínicos

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-07-15
Última Modificação: 2025-09-12

Descrição:
    Este módulo faz parte do framework multi-paradigma desenvolvido para predição
    de abandono de tratamento em pacientes com tuberculose. Implementa técnicas
    de NLP para extrair features de narrativas clínicas, incluindo TF-IDF,
    embeddings e análise de sentimentos.

Licença: MIT
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """
    Extrator de features de textos clínicos.
    
    Implementa múltiplas técnicas de NLP:
    - TF-IDF (Term Frequency-Inverse Document Frequency)
    - Bag of Words
    - Topic Modeling (LDA)
    - Estatísticas textuais
    
    Attributes:
        tfidf_vectorizer: Vetorizador TF-IDF
        count_vectorizer: Vetorizador Bag of Words
        lda_model: Modelo LDA para topic modeling
        output_dir: Diretório para salvar modelos
    """
    
    def __init__(
        self,
        max_features: int = 100,
        n_topics: int = 10,
        output_dir: str = 'results/nlp/features'
    ):
        """
        Inicializa o extrator de features.
        
        Args:
            max_features: Número máximo de features TF-IDF
            n_topics: Número de tópicos para LDA
            output_dir: Diretório para salvar resultados
        """
        self.max_features = max_features
        self.n_topics = n_topics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar vetorizadores
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',  # Usar stopwords em português na prática
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        
        logger.info(f'TextFeatureExtractor inicializado (max_features={max_features}, n_topics={n_topics})')
    
    def extract_tfidf_features(
        self,
        texts: List[str],
        fit: bool = True
    ) -> np.ndarray:
        """
        Extrai features TF-IDF dos textos.
        
        Args:
            texts: Lista de textos
            fit: Se True, treina o vetorizador; se False, apenas transforma
        
        Returns:
            Matriz de features TF-IDF
        """
        logger.info(f'Extraindo features TF-IDF de {len(texts)} textos')
        
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f'Vocabulário TF-IDF: {len(self.tfidf_vectorizer.vocabulary_)} termos')
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix.toarray()
    
    def extract_bow_features(
        self,
        texts: List[str],
        fit: bool = True
    ) -> np.ndarray:
        """
        Extrai features Bag of Words dos textos.
        
        Args:
            texts: Lista de textos
            fit: Se True, treina o vetorizador; se False, apenas transforma
        
        Returns:
            Matriz de features BoW
        """
        logger.info(f'Extraindo features Bag of Words de {len(texts)} textos')
        
        if fit:
            bow_matrix = self.count_vectorizer.fit_transform(texts)
        else:
            bow_matrix = self.count_vectorizer.transform(texts)
        
        return bow_matrix.toarray()
    
    def extract_topic_features(
        self,
        texts: List[str],
        fit: bool = True
    ) -> np.ndarray:
        """
        Extrai features de tópicos usando LDA.
        
        Args:
            texts: Lista de textos
            fit: Se True, treina o modelo LDA; se False, apenas transforma
        
        Returns:
            Matriz de distribuição de tópicos
        """
        logger.info(f'Extraindo features de tópicos (LDA) de {len(texts)} textos')
        
        # Primeiro extrair BoW
        bow_matrix = self.extract_bow_features(texts, fit=fit)
        
        if fit:
            topic_matrix = self.lda_model.fit_transform(bow_matrix)
            logger.info(f'Modelo LDA treinado com {self.n_topics} tópicos')
        else:
            topic_matrix = self.lda_model.transform(bow_matrix)
        
        return topic_matrix
    
    def extract_text_statistics(self, texts: List[str]) -> pd.DataFrame:
        """
        Extrai estatísticas textuais básicas.
        
        Args:
            texts: Lista de textos
        
        Returns:
            DataFrame com estatísticas
        """
        logger.info(f'Extraindo estatísticas textuais de {len(texts)} textos')
        
        stats = []
        for text in texts:
            if pd.isna(text) or text == '':
                stats.append({
                    'text_length': 0,
                    'word_count': 0,
                    'avg_word_length': 0,
                    'sentence_count': 0
                })
            else:
                words = text.split()
                sentences = text.split('.')
                
                stats.append({
                    'text_length': len(text),
                    'word_count': len(words),
                    'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                    'sentence_count': len([s for s in sentences if s.strip()])
                })
        
        return pd.DataFrame(stats)
    
    def extract_all_features(
        self,
        texts: List[str],
        fit: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extrai todas as features de texto.
        
        Args:
            texts: Lista de textos
            fit: Se True, treina os modelos; se False, apenas transforma
        
        Returns:
            Tupla (features_numéricas, features_estatísticas)
        """
        logger.info(f'Extraindo todas as features de {len(texts)} textos')
        
        # TF-IDF
        tfidf_features = self.extract_tfidf_features(texts, fit=fit)
        
        # Topic modeling
        topic_features = self.extract_topic_features(texts, fit=fit)
        
        # Estatísticas
        stats_features = self.extract_text_statistics(texts)
        
        # Concatenar features numéricas
        numeric_features = np.hstack([tfidf_features, topic_features])
        
        logger.info(f'Total de features extraídas: {numeric_features.shape[1]} numéricas + {stats_features.shape[1]} estatísticas')
        
        return numeric_features, stats_features
    
    def get_top_terms(self, n_terms: int = 20) -> List[str]:
        """
        Retorna os termos mais importantes do vocabulário TF-IDF.
        
        Args:
            n_terms: Número de termos a retornar
        
        Returns:
            Lista de termos mais importantes
        """
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            logger.warning('Vetorizador TF-IDF não foi treinado ainda')
            return []
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return list(feature_names[:n_terms])
    
    def get_topic_words(self, n_words: int = 10) -> Dict[int, List[str]]:
        """
        Retorna as palavras mais importantes de cada tópico LDA.
        
        Args:
            n_words: Número de palavras por tópico
        
        Returns:
            Dicionário {topic_id: [palavras]}
        """
        if not hasattr(self.lda_model, 'components_'):
            logger.warning('Modelo LDA não foi treinado ainda')
            return {}
        
        feature_names = self.count_vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            topics[topic_idx] = [feature_names[i] for i in top_indices]
        
        logger.info(f'Palavras principais extraídas para {len(topics)} tópicos')
        
        return topics
    
    def save_models(self) -> None:
        """Salva os modelos treinados."""
        logger.info('Salvando modelos de NLP')
        
        joblib.dump(
            self.tfidf_vectorizer,
            self.output_dir / 'tfidf_vectorizer.pkl'
        )
        
        joblib.dump(
            self.count_vectorizer,
            self.output_dir / 'count_vectorizer.pkl'
        )
        
        joblib.dump(
            self.lda_model,
            self.output_dir / 'lda_model.pkl'
        )
        
        logger.info(f'Modelos salvos em {self.output_dir}')
    
    def load_models(self) -> None:
        """Carrega modelos previamente treinados."""
        logger.info('Carregando modelos de NLP')
        
        self.tfidf_vectorizer = joblib.load(
            self.output_dir / 'tfidf_vectorizer.pkl'
        )
        
        self.count_vectorizer = joblib.load(
            self.output_dir / 'count_vectorizer.pkl'
        )
        
        self.lda_model = joblib.load(
            self.output_dir / 'lda_model.pkl'
        )
        
        logger.info('Modelos carregados com sucesso')
