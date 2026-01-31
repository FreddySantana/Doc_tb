"""
Módulo de Métricas de Interpretabilidade

Autor: Frederico Guilheme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-07-10
Última Modificação: 2025-11-25

Descrição:
    Implementa métricas formais para quantificar a interpretabilidade de modelos
    de machine learning, combinando SHAP e LIME conforme a tese.
    
    Métrica de Interpretabilidade (0.0 a 1.0):
    - Baseada em consistência entre SHAP e LIME
    - Baseada em cobertura de features importantes
    - Baseada em estabilidade das explicações

Licença: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InterpretabilityMetrics:
    """Classe para armazenar métricas de interpretabilidade"""
    consistency_score: float  # Consistência entre SHAP e LIME
    coverage_score: float     # Cobertura de features importantes
    stability_score: float    # Estabilidade das explicações
    overall_score: float      # Score geral (0.0 a 1.0)


class InterpretabilityCalculator:
    """
    Calcula métricas formais de interpretabilidade.
    
    A métrica geral de interpretabilidade é uma combinação ponderada de:
    1. Consistência entre SHAP e LIME (40%)
    2. Cobertura de features importantes (35%)
    3. Estabilidade das explicações (25%)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Inicializa o calculador de interpretabilidade.
        
        Parâmetros:
        -----------
        weights : Dict[str, float], optional
            Pesos para cada componente da métrica
        """
        if weights is None:
            self.weights = {
                'consistency': 0.40,
                'coverage': 0.35,
                'stability': 0.25
            }
        else:
            self.weights = weights
        
        logger.info(f"✅ InterpretabilityCalculator inicializado")
        logger.info(f"   Pesos: {self.weights}")
    
    def calculate_consistency_score(
        self,
        shap_values: np.ndarray,
        lime_values: np.ndarray,
        method: str = 'correlation'
    ) -> float:
        """
        Calcula score de consistência entre SHAP e LIME.
        
        Parâmetros:
        -----------
        shap_values : np.ndarray
            Valores SHAP (shape: n_samples, n_features)
        lime_values : np.ndarray
            Valores LIME (shape: n_samples, n_features)
        method : str
            Método de cálculo ('correlation', 'ranking', 'agreement')
            
        Retorna:
        --------
        float
            Score de consistência (0.0 a 1.0)
        """
        # Normalizar valores
        shap_norm = np.abs(shap_values)
        lime_norm = np.abs(lime_values)
        
        # Normalizar para [0, 1]
        shap_norm = (shap_norm - shap_norm.min(axis=0)) / (shap_norm.max(axis=0) - shap_norm.min(axis=0) + 1e-10)
        lime_norm = (lime_norm - lime_norm.min(axis=0)) / (lime_norm.max(axis=0) - lime_norm.min(axis=0) + 1e-10)
        
        if method == 'correlation':
            # Correlação de Pearson entre SHAP e LIME
            correlations = []
            for i in range(shap_norm.shape[0]):
                corr = np.corrcoef(shap_norm[i], lime_norm[i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            consistency = np.mean(correlations) if correlations else 0.0
            consistency = max(0.0, consistency)  # Garantir [0, 1]
            
        elif method == 'ranking':
            # Concordância no ranking de features importantes
            shap_ranking = np.argsort(-np.abs(shap_values).mean(axis=0))
            lime_ranking = np.argsort(-np.abs(lime_values).mean(axis=0))
            
            # Spearman correlation
            n = len(shap_ranking)
            d = np.sum((shap_ranking - lime_ranking)**2)
            consistency = 1 - (6 * d) / (n * (n**2 - 1))
            consistency = max(0.0, consistency)
            
        elif method == 'agreement':
            # Percentual de concordância nos top-k features
            k = max(1, int(0.2 * shap_norm.shape[1]))  # Top 20% das features
            
            shap_top = set(np.argsort(-np.abs(shap_values).mean(axis=0))[:k])
            lime_top = set(np.argsort(-np.abs(lime_values).mean(axis=0))[:k])
            
            agreement = len(shap_top & lime_top) / k
            consistency = agreement
            
        else:
            raise ValueError(f"Método inválido: {method}")
        
        return float(consistency)
    
    def calculate_coverage_score(
        self,
        shap_values: np.ndarray,
        lime_values: np.ndarray,
        threshold: float = 0.1
    ) -> float:
        """
        Calcula score de cobertura de features importantes.
        
        Parâmetros:
        -----------
        shap_values : np.ndarray
            Valores SHAP (shape: n_samples, n_features)
        lime_values : np.ndarray
            Valores LIME (shape: n_samples, n_features)
        threshold : float
            Limiar para considerar uma feature importante
            
        Retorna:
        --------
        float
            Score de cobertura (0.0 a 1.0)
        """
        # Calcular importância média
        shap_importance = np.abs(shap_values).mean(axis=0)
        lime_importance = np.abs(lime_values).mean(axis=0)
        
        # Normalizar
        shap_importance = shap_importance / (shap_importance.max() + 1e-10)
        lime_importance = lime_importance / (lime_importance.max() + 1e-10)
        
        # Features importantes (acima do threshold)
        shap_important = np.sum(shap_importance > threshold)
        lime_important = np.sum(lime_importance > threshold)
        
        # Cobertura: proporção de features importantes identificadas
        total_features = shap_values.shape[1]
        coverage = (shap_important + lime_important) / (2 * total_features)
        
        return float(coverage)
    
    def calculate_stability_score(
        self,
        shap_values_list: List[np.ndarray],
        lime_values_list: List[np.ndarray]
    ) -> float:
        """
        Calcula score de estabilidade das explicações.
        
        Parâmetros:
        -----------
        shap_values_list : List[np.ndarray]
            Lista de valores SHAP de múltiplas execuções
        lime_values_list : List[np.ndarray]
            Lista de valores LIME de múltiplas execuções
            
        Retorna:
        --------
        float
            Score de estabilidade (0.0 a 1.0)
        """
        if len(shap_values_list) < 2 or len(lime_values_list) < 2:
            logger.warning("Necessário pelo menos 2 execuções para calcular estabilidade")
            return 1.0
        
        # Calcular variância das explicações
        shap_var = np.var([np.abs(s).mean(axis=0) for s in shap_values_list])
        lime_var = np.var([np.abs(l).mean(axis=0) for l in lime_values_list])
        
        # Estabilidade: inverso da variância normalizada
        max_var = 1.0
        stability = 1.0 - (shap_var + lime_var) / (2 * max_var)
        stability = max(0.0, min(1.0, stability))
        
        return float(stability)
    
    def calculate_overall_score(
        self,
        shap_values: np.ndarray,
        lime_values: np.ndarray,
        shap_values_list: Optional[List[np.ndarray]] = None,
        lime_values_list: Optional[List[np.ndarray]] = None
    ) -> InterpretabilityMetrics:
        """
        Calcula score geral de interpretabilidade.
        
        Parâmetros:
        -----------
        shap_values : np.ndarray
            Valores SHAP
        lime_values : np.ndarray
            Valores LIME
        shap_values_list : List[np.ndarray], optional
            Lista de valores SHAP para calcular estabilidade
        lime_values_list : List[np.ndarray], optional
            Lista de valores LIME para calcular estabilidade
            
        Retorna:
        --------
        InterpretabilityMetrics
            Métricas de interpretabilidade
        """
        # Calcular componentes
        consistency = self.calculate_consistency_score(shap_values, lime_values)
        coverage = self.calculate_coverage_score(shap_values, lime_values)
        
        if shap_values_list is not None and lime_values_list is not None:
            stability = self.calculate_stability_score(shap_values_list, lime_values_list)
        else:
            stability = 1.0  # Assumir estabilidade máxima se não houver múltiplas execuções
        
        # Score geral (combinação ponderada)
        overall = (
            self.weights['consistency'] * consistency +
            self.weights['coverage'] * coverage +
            self.weights['stability'] * stability
        )
        
        return InterpretabilityMetrics(
            consistency_score=consistency,
            coverage_score=coverage,
            stability_score=stability,
            overall_score=overall
        )
    
    def calculate_per_sample_interpretability(
        self,
        shap_values: np.ndarray,
        lime_values: np.ndarray
    ) -> np.ndarray:
        """
        Calcula score de interpretabilidade por amostra.
        
        Parâmetros:
        -----------
        shap_values : np.ndarray
            Valores SHAP (shape: n_samples, n_features)
        lime_values : np.ndarray
            Valores LIME (shape: n_samples, n_features)
            
        Retorna:
        --------
        np.ndarray
            Score de interpretabilidade por amostra (shape: n_samples)
        """
        n_samples = shap_values.shape[0]
        interpretability_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Correlação entre SHAP e LIME para esta amostra
            shap_sample = np.abs(shap_values[i])
            lime_sample = np.abs(lime_values[i])
            
            # Normalizar
            shap_sample = (shap_sample - shap_sample.min()) / (shap_sample.max() - shap_sample.min() + 1e-10)
            lime_sample = (lime_sample - lime_sample.min()) / (lime_sample.max() - lime_sample.min() + 1e-10)
            
            # Correlação
            corr = np.corrcoef(shap_sample, lime_sample)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            
            # Score: média entre correlação e concordância
            top_k = max(1, int(0.2 * len(shap_sample)))
            shap_top = set(np.argsort(-shap_sample)[:top_k])
            lime_top = set(np.argsort(-lime_sample)[:top_k])
            agreement = len(shap_top & lime_top) / top_k
            
            interpretability_scores[i] = (max(0.0, corr) + agreement) / 2.0
        
        return interpretability_scores
    
    def generate_report(
        self,
        shap_values: np.ndarray,
        lime_values: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """
        Gera relatório de interpretabilidade.
        
        Parâmetros:
        -----------
        shap_values : np.ndarray
            Valores SHAP
        lime_values : np.ndarray
            Valores LIME
        model_name : str
            Nome do modelo
            
        Retorna:
        --------
        str
            Relatório formatado
        """
        metrics = self.calculate_overall_score(shap_values, lime_values)
        
        report = f"""
{'='*60}
RELATÓRIO DE INTERPRETABILIDADE - {model_name}
{'='*60}

Métrica de Consistência (SHAP vs LIME):
  Score: {metrics.consistency_score:.4f}
  Interpretação: {'Muito consistentes' if metrics.consistency_score > 0.7 else 'Moderadamente consistentes' if metrics.consistency_score > 0.4 else 'Pouco consistentes'}

Métrica de Cobertura (Features Importantes):
  Score: {metrics.coverage_score:.4f}
  Interpretação: {'Excelente cobertura' if metrics.coverage_score > 0.7 else 'Boa cobertura' if metrics.coverage_score > 0.4 else 'Cobertura limitada'}

Métrica de Estabilidade (Explicações):
  Score: {metrics.stability_score:.4f}
  Interpretação: {'Muito estável' if metrics.stability_score > 0.7 else 'Moderadamente estável' if metrics.stability_score > 0.4 else 'Pouco estável'}

SCORE GERAL DE INTERPRETABILIDADE:
  {metrics.overall_score:.4f}
  
  Classificação:
  - 0.90-1.00: Excelente interpretabilidade
  - 0.70-0.89: Boa interpretabilidade
  - 0.50-0.69: Interpretabilidade moderada
  - 0.30-0.49: Interpretabilidade limitada
  - 0.00-0.29: Interpretabilidade muito limitada

{'='*60}
"""
        return report


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados sintéticos
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Simular SHAP e LIME
    shap_values = np.random.randn(n_samples, n_features)
    lime_values = shap_values + np.random.randn(n_samples, n_features) * 0.2
    
    # Calcular interpretabilidade
    calc = InterpretabilityCalculator()
    metrics = calc.calculate_overall_score(shap_values, lime_values)
    
    print(calc.generate_report(shap_values, lime_values, "LightGBM"))
    
    # Interpretabilidade por amostra
    per_sample = calc.calculate_per_sample_interpretability(shap_values, lime_values)
    print(f"\nInterpretabilidade por amostra (média): {per_sample.mean():.4f}")
    print(f"Interpretabilidade por amostra (std): {per_sample.std():.4f}")
