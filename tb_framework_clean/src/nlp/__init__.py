"""
Módulo de Natural Language Processing (NLP)

Implementa processamento de texto clínico usando BioBERT
conforme descrito na Seção 4.4 da tese.
"""

from .text_processor import TextProcessor
from .train_biobert import BioBERTModel, train_biobert
from .nlp_pipeline import NLPPipeline

__all__ = [
    'TextProcessor',
    'BioBERTModel',
    'train_biobert',
    'NLPPipeline'
]
