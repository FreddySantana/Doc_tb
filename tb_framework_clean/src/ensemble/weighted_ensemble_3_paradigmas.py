"""
Módulo de Ensemble Ponderado com 3 Paradigmas (ML + DRL + NLP)

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-08-20
Última Modificação: 2025-11-25

Descrição:
    Este módulo faz parte do framework multi-paradigma desenvolvido para predição
    de abandono de tratamento em pacientes com tuberculose. O framework integra
    técnicas de Machine Learning, Deep Reinforcement Learning, Natural Language
    Processing e Explainable AI.

Licença: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import joblib
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedEnsemble3Paradigms:
    """
    Ensemble ponderado que combina 3 paradigmas de IA.
    
    Equação:
    --------
    p̂_ensemble(x) = w_ML × p̂_ML(x) + w_DRL × p̂_DRL(x) + w_NLP × p̂_NLP(x)
    
    Onde:
    - w_ML + w_DRL + w_NLP = 1.0
    - p̂_i(x) ∈ [0, 1] (probabilidade de abandono)
    
    Atributos:
    ----------
    weights : Dict[str, float]
        Pesos otimizados para cada paradigma
    threshold : float
        Limiar de classificação otimizado
    """
    
    def __init__(self):
        """Inicializa o ensemble com pesos padrão."""
        self.weights = {
            'ml': 0.50,   # Peso inicial para ML
            'drl': 0.30,  # Peso inicial para DRL
            'nlp': 0.20   # Peso inicial para NLP
        }
        self.threshold = 0.5
        self.is_fitted = False
        
    def predict_proba(
        self,
        ml_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray
    ) -> np.ndarray:
        """
        Calcula a probabilidade final do ensemble.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
            
        Retorna:
        --------
        np.ndarray
            Probabilidades finais do ensemble
        """
        # Validar dimensões
        assert len(ml_proba) == len(drl_proba) == len(nlp_proba), \
            "Todas as predições devem ter o mesmo tamanho"
        
        # Calcular ensemble ponderado
        ensemble_proba = (
            self.weights['ml'] * ml_proba +
            self.weights['drl'] * drl_proba +
            self.weights['nlp'] * nlp_proba
        )
        
        return ensemble_proba
    
    def predict(
        self,
        ml_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray
    ) -> np.ndarray:
        """
        Prediz classes usando o limiar otimizado.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
            
        Retorna:
        --------
        np.ndarray
            Classes preditas (0 ou 1)
        """
        proba = self.predict_proba(ml_proba, drl_proba, nlp_proba)
        return (proba >= self.threshold).astype(int)
    
    def optimize_weights(
        self,
        ml_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        y_true: np.ndarray,
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """
        Otimiza os pesos do ensemble usando Grid Search.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        y_true : np.ndarray
            Labels verdadeiros
        metric : str
            Métrica a otimizar ('f1', 'auc', 'accuracy')
            
        Retorna:
        --------
        Dict[str, float]
            Pesos otimizados
        """
        logger.info("Iniciando otimização de pesos do ensemble...")
        
        # Grid de pesos possíveis
        weight_grid = np.arange(0.0, 1.05, 0.05)
        
        best_score = -np.inf
        best_weights = self.weights.copy()
        
        # Grid Search
        total_combinations = 0
        for w_ml in weight_grid:
            for w_drl in weight_grid:
                w_nlp = 1.0 - w_ml - w_drl
                
                # Verificar se os pesos são válidos
                if w_nlp < 0 or w_nlp > 1:
                    continue
                
                total_combinations += 1
                
                # Calcular predições com esses pesos
                temp_weights = {'ml': w_ml, 'drl': w_drl, 'nlp': w_nlp}
                ensemble_proba = (
                    w_ml * ml_proba +
                    w_drl * drl_proba +
                    w_nlp * nlp_proba
                )
                
                # Calcular métrica
                y_pred = (ensemble_proba >= self.threshold).astype(int)
                
                if metric == 'f1':
                    score = f1_score(y_true, y_pred)
                elif metric == 'auc':
                    score = roc_auc_score(y_true, ensemble_proba)
                elif metric == 'accuracy':
                    score = accuracy_score(y_true, y_pred)
                else:
                    raise ValueError(f"Métrica inválida: {metric}")
                
                # Atualizar melhor configuração
                if score > best_score:
                    best_score = score
                    best_weights = temp_weights.copy()
        
        logger.info(f"Grid Search completo: {total_combinations} combinações testadas")
        logger.info(f"Melhor {metric}: {best_score:.4f}")
        logger.info(f"Melhores pesos: ML={best_weights['ml']:.2f}, "
                   f"DRL={best_weights['drl']:.2f}, NLP={best_weights['nlp']:.2f}")
        
        # Atualizar pesos
        self.weights = best_weights
        self.is_fitted = True
        
        return best_weights
    
    def optimize_threshold(
        self,
        ml_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        y_true: np.ndarray,
        metric: str = 'f1'
    ) -> float:
        """
        Otimiza o limiar de classificação.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        y_true : np.ndarray
            Labels verdadeiros
        metric : str
            Métrica a otimizar ('f1', 'accuracy')
            
        Retorna:
        --------
        float
            Limiar otimizado
        """
        logger.info("Iniciando otimização de limiar...")
        
        # Calcular probabilidades do ensemble
        ensemble_proba = self.predict_proba(ml_proba, drl_proba, nlp_proba)
        
        # Testar diferentes limiares
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (ensemble_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"Métrica inválida: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Melhor limiar: {best_threshold:.2f} ({metric}={best_score:.4f})")
        
        # Atualizar limiar
        self.threshold = best_threshold
        
        return best_threshold
    
    def fit(
        self,
        ml_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        y_true: np.ndarray,
        optimize_weights: bool = True,
        optimize_threshold: bool = True,
        metric: str = 'f1'
    ):
        """
        Treina o ensemble otimizando pesos e limiar.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        y_true : np.ndarray
            Labels verdadeiros
        optimize_weights : bool
            Se True, otimiza os pesos
        optimize_threshold : bool
            Se True, otimiza o limiar
        metric : str
            Métrica a otimizar
        """
        if optimize_weights:
            self.optimize_weights(ml_proba, drl_proba, nlp_proba, y_true, metric)
        
        if optimize_threshold:
            self.optimize_threshold(ml_proba, drl_proba, nlp_proba, y_true, metric)
        
        self.is_fitted = True
        logger.info("Ensemble treinado com sucesso!")
    
    def evaluate(
        self,
        ml_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Avalia o desempenho do ensemble.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        y_true : np.ndarray
            Labels verdadeiros
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de desempenho
        """
        # Predições
        ensemble_proba = self.predict_proba(ml_proba, drl_proba, nlp_proba)
        y_pred = self.predict(ml_proba, drl_proba, nlp_proba)
        
        # Calcular métricas
        metrics = {
            'f1_score': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, ensemble_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'weights_ml': self.weights['ml'],
            'weights_drl': self.weights['drl'],
            'weights_nlp': self.weights['nlp'],
            'threshold': self.threshold
        }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Salva o ensemble treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho para salvar o modelo
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'weights': self.weights,
            'threshold': self.threshold,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble salvo em: {filepath}")
    
    def load(self, filepath: str):
        """
        Carrega um ensemble treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho do modelo salvo
        """
        model_data = joblib.load(filepath)
        
        self.weights = model_data['weights']
        self.threshold = model_data['threshold']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Ensemble carregado de: {filepath}")
        logger.info(f"Pesos: ML={self.weights['ml']:.2f}, "
                   f"DRL={self.weights['drl']:.2f}, NLP={self.weights['nlp']:.2f}")


def main():
    """Exemplo de uso do ensemble."""
    # Dados de exemplo
    np.random.seed(42)
    n_samples = 1000
    
    # Simular probabilidades dos 3 paradigmas
    ml_proba = np.random.rand(n_samples)
    drl_proba = np.random.rand(n_samples)
    nlp_proba = np.random.rand(n_samples)
    
    # Labels verdadeiros
    y_true = np.random.randint(0, 2, n_samples)
    
    # Criar e treinar ensemble
    ensemble = WeightedEnsemble3Paradigms()
    ensemble.fit(ml_proba, drl_proba, nlp_proba, y_true)
    
    # Avaliar
    metrics = ensemble.evaluate(ml_proba, drl_proba, nlp_proba, y_true)
    
    print("\n=== Resultados do Ensemble ===")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nPesos Otimizados:")
    print(f"  ML:  {metrics['weights_ml']:.2f}")
    print(f"  DRL: {metrics['weights_drl']:.2f}")
    print(f"  NLP: {metrics['weights_nlp']:.2f}")
    print(f"\nLimiar: {metrics['threshold']:.2f}")
    
    # Salvar
    ensemble.save('results/ensemble/ensemble_3_paradigmas.pkl')


if __name__ == "__main__":
    main()
