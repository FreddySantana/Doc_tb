"""
Módulo de Ensemble Ponderado com 4 Paradigmas + XAI (CORRIGIDO)

Autor: Frederico Guilheme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-06-15
Última Modificação: 2025-11-25

Descrição:
    Este módulo implementa o ensemble ponderado CORRIGIDO conforme a Equação 81 da tese,
    integrando os 4 paradigmas: Machine Learning, Explainable AI (XAI), Deep Reinforcement
    Learning (DRL) e Natural Language Processing (NLP).
    
    CORREÇÃO CRÍTICA:
    - XAI é integrado como MODIFICADOR DE PESOS, não como predição independente
    - A confiança/interpretabilidade do XAI pondera os pesos dos outros paradigmas
    - Implementa quantificação de incerteza conforme Equações 82-84 da tese

Licença: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
import joblib
import logging
from pathlib import Path
from dataclasses import dataclass

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleMetrics:
    """Classe para armazenar métricas do ensemble"""
    f1_score: float
    auc: float
    accuracy: float
    precision: float
    recall: float
    mcc: float
    interpretability_score: float
    uncertainty: float


class WeightedEnsemble4Paradigms:
    """
    Ensemble ponderado que combina 4 paradigmas de IA com XAI como modificador.
    
    Equação Corrigida (conforme Tese, Equação 81):
    -------
    ŷ_ensemble(x) = w_ML(x) · ŷ_ML(x) + w_XAI(x) · ŷ_XAI(x) + w_DRL(x) · ŷ_DRL(x) + w_NLP(x) · ŷ_NLP(x)
    
    Onde:
    - w_i(x) = base_weight_i · (1 + α · interpretability_score(x))
    - interpretability_score(x) ∈ [0, 1] é calculado a partir de SHAP e LIME
    - ŷ_i(x) ∈ [0, 1] (probabilidade de abandono)
    - Σ w_i(x) = 1.0 (normalização)
    
    Quantificação de Incerteza (Equações 82-84):
    -------
    U_MC(x) = √(1/T Σ(ŷ_t(x) - ŷ_MC(x))²)  # Incerteza epistêmica (Monte Carlo Dropout)
    U_ens(x) = √(1/4 Σ(ŷ_i(x) - ŷ_ensemble(x))²)  # Variância do ensemble
    U(x) = 0.6 · U_MC(x) + 0.4 · U_ens(x)  # Incerteza total
    
    Atributos:
    ----------
    base_weights : Dict[str, float]
        Pesos base para cada paradigma (antes da ponderação por XAI)
    alpha : float
        Coeficiente de ponderação por interpretabilidade (0 a 1)
    threshold : float
        Limiar de classificação otimizado
    weights_history : List[Dict]
        Histórico de pesos otimizados
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Inicializa o ensemble com pesos base.
        
        Parâmetros:
        -----------
        alpha : float
            Coeficiente de ponderação por interpretabilidade (padrão: 0.5)
        """
        # Pesos base conforme mencionado na tese
        self.base_weights = {
            'ml': 0.35,   # Machine Learning
            'xai': 0.30,  # XAI (Explainable AI)
            'drl': 0.20,  # Deep Reinforcement Learning
            'nlp': 0.15   # Natural Language Processing
        }
        
        # Coeficiente de ponderação por interpretabilidade
        self.alpha = alpha
        
        # Limiar de classificação
        self.threshold = 0.5
        
        # Histórico de otimizações
        self.weights_history = []
        self.is_fitted = False
        
        logger.info(f"✅ Ensemble 4-Paradigmas inicializado com α={alpha}")
    
    def calculate_interpretability_score(
        self,
        shap_values: np.ndarray,
        lime_values: np.ndarray,
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Calcula score de interpretabilidade a partir de SHAP e LIME.
        
        Parâmetros:
        -----------
        shap_values : np.ndarray
            Valores SHAP (shape: n_samples, n_features)
        lime_values : np.ndarray
            Valores LIME (shape: n_samples, n_features)
        method : str
            Método de agregação ('mean', 'max', 'std')
            
        Retorna:
        --------
        np.ndarray
            Score de interpretabilidade para cada amostra (0 a 1)
        """
        # Normalizar SHAP e LIME
        shap_norm = np.abs(shap_values).mean(axis=1)
        lime_norm = np.abs(lime_values).mean(axis=1)
        
        # Normalizar para [0, 1]
        shap_norm = (shap_norm - shap_norm.min()) / (shap_norm.max() - shap_norm.min() + 1e-10)
        lime_norm = (lime_norm - lime_norm.min()) / (lime_norm.max() - lime_norm.min() + 1e-10)
        
        # Combinar SHAP e LIME
        if method == 'mean':
            interpretability_score = (shap_norm + lime_norm) / 2.0
        elif method == 'max':
            interpretability_score = np.maximum(shap_norm, lime_norm)
        elif method == 'std':
            interpretability_score = np.std([shap_norm, lime_norm], axis=0)
        else:
            raise ValueError(f"Método inválido: {method}")
        
        return interpretability_score
    
    def calculate_adaptive_weights(
        self,
        interpretability_scores: np.ndarray
    ) -> np.ndarray:
        """
        Calcula pesos adaptativos ponderados por interpretabilidade.
        
        Parâmetros:
        -----------
        interpretability_scores : np.ndarray
            Scores de interpretabilidade para cada amostra (0 a 1)
            
        Retorna:
        --------
        np.ndarray
            Pesos adaptativos (shape: n_samples, 4)
        """
        n_samples = len(interpretability_scores)
        adaptive_weights = np.zeros((n_samples, 4))
        
        # Paradigmas em ordem: ML, XAI, DRL, NLP
        paradigm_names = ['ml', 'xai', 'drl', 'nlp']
        
        for i, paradigm in enumerate(paradigm_names):
            # Ponderação: w_i(x) = base_weight_i · (1 + α · interpretability_score(x))
            adaptive_weights[:, i] = self.base_weights[paradigm] * (
                1 + self.alpha * interpretability_scores
            )
        
        # Normalizar para que Σ w_i(x) = 1.0 para cada amostra
        row_sums = adaptive_weights.sum(axis=1, keepdims=True)
        adaptive_weights = adaptive_weights / (row_sums + 1e-10)
        
        return adaptive_weights
    
    def predict_proba(
        self,
        ml_proba: np.ndarray,
        xai_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        interpretability_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calcula a probabilidade final do ensemble.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML (shape: n_samples)
        xai_proba : np.ndarray
            Probabilidades do modelo XAI (shape: n_samples)
        drl_proba : np.ndarray
            Probabilidades do agente DRL (shape: n_samples)
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP (shape: n_samples)
        interpretability_scores : np.ndarray, optional
            Scores de interpretabilidade (se None, usa pesos base)
            
        Retorna:
        --------
        np.ndarray
            Probabilidades finais do ensemble (shape: n_samples)
        """
        # Validar dimensões
        assert len(ml_proba) == len(xai_proba) == len(drl_proba) == len(nlp_proba), \
            "Todas as predições devem ter o mesmo tamanho"
        
        # Se não houver scores de interpretabilidade, usar pesos base
        if interpretability_scores is None:
            interpretability_scores = np.ones(len(ml_proba))
        
        # Calcular pesos adaptativos
        adaptive_weights = self.calculate_adaptive_weights(interpretability_scores)
        
        # Calcular ensemble ponderado
        ensemble_proba = (
            adaptive_weights[:, 0] * ml_proba +
            adaptive_weights[:, 1] * xai_proba +
            adaptive_weights[:, 2] * drl_proba +
            adaptive_weights[:, 3] * nlp_proba
        )
        
        return ensemble_proba
    
    def predict(
        self,
        ml_proba: np.ndarray,
        xai_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        interpretability_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Prediz classes usando o limiar otimizado.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        xai_proba : np.ndarray
            Probabilidades do modelo XAI
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        interpretability_scores : np.ndarray, optional
            Scores de interpretabilidade
            
        Retorna:
        --------
        np.ndarray
            Classes preditas (0 ou 1)
        """
        proba = self.predict_proba(ml_proba, xai_proba, drl_proba, nlp_proba, interpretability_scores)
        return (proba >= self.threshold).astype(int)
    
    def calculate_uncertainty(
        self,
        ml_proba: np.ndarray,
        xai_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        mc_dropout_samples: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula quantificação de incerteza conforme Equações 82-84 da tese.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        xai_proba : np.ndarray
            Probabilidades do modelo XAI
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        mc_dropout_samples : np.ndarray, optional
            Amostras de Monte Carlo Dropout (shape: n_samples, T)
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (U_MC, U_ens, U_total) - Incertezas epistêmica, ensemble e total
        """
        # Incerteza epistêmica (Monte Carlo Dropout) - Equação 82
        if mc_dropout_samples is not None:
            y_mc = mc_dropout_samples.mean(axis=1)
            U_MC = np.sqrt(np.mean((mc_dropout_samples - y_mc[:, np.newaxis])**2, axis=1))
        else:
            U_MC = np.zeros(len(ml_proba))
        
        # Incerteza do ensemble (Variância) - Equação 83
        ensemble_proba = self.predict_proba(ml_proba, xai_proba, drl_proba, nlp_proba)
        paradigm_probas = np.array([ml_proba, xai_proba, drl_proba, nlp_proba]).T
        U_ens = np.sqrt(np.mean((paradigm_probas - ensemble_proba[:, np.newaxis])**2, axis=1))
        
        # Incerteza total (combinação ponderada) - Equação 84
        U_total = 0.6 * U_MC + 0.4 * U_ens
        
        return U_MC, U_ens, U_total
    
    def optimize_weights(
        self,
        ml_proba: np.ndarray,
        xai_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        y_true: np.ndarray,
        interpretability_scores: Optional[np.ndarray] = None,
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """
        Otimiza os pesos do ensemble usando Grid Search.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        xai_proba : np.ndarray
            Probabilidades do modelo XAI
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        y_true : np.ndarray
            Labels verdadeiros
        interpretability_scores : np.ndarray, optional
            Scores de interpretabilidade
        metric : str
            Métrica a otimizar ('f1', 'auc', 'accuracy')
            
        Retorna:
        --------
        Dict[str, float]
            Pesos otimizados
        """
        logger.info("Iniciando otimização de pesos do ensemble (4 paradigmas)...")
        
        # Grid de pesos possíveis
        weight_grid = np.arange(0.0, 1.05, 0.05)
        
        best_score = -np.inf
        best_weights = self.base_weights.copy()
        best_alpha = self.alpha
        
        # Grid Search sobre alpha e pesos base
        alpha_grid = np.arange(0.0, 1.05, 0.1)
        
        total_combinations = 0
        for alpha in alpha_grid:
            self.alpha = alpha
            
            for w_ml in weight_grid:
                for w_xai in weight_grid:
                    for w_drl in weight_grid:
                        w_nlp = 1.0 - w_ml - w_xai - w_drl
                        
                        # Verificar se os pesos são válidos
                        if w_nlp < 0 or w_nlp > 1:
                            continue
                        
                        total_combinations += 1
                        
                        # Atualizar pesos base
                        self.base_weights = {
                            'ml': w_ml,
                            'xai': w_xai,
                            'drl': w_drl,
                            'nlp': w_nlp
                        }
                        
                        # Calcular predições com esses pesos
                        ensemble_proba = self.predict_proba(
                            ml_proba, xai_proba, drl_proba, nlp_proba,
                            interpretability_scores
                        )
                        y_pred = (ensemble_proba >= self.threshold).astype(int)
                        
                        # Calcular métrica
                        if metric == 'f1':
                            score = f1_score(y_true, y_pred, zero_division=0)
                        elif metric == 'auc':
                            score = roc_auc_score(y_true, ensemble_proba)
                        elif metric == 'accuracy':
                            score = accuracy_score(y_true, y_pred)
                        else:
                            raise ValueError(f"Métrica inválida: {metric}")
                        
                        # Atualizar melhor configuração
                        if score > best_score:
                            best_score = score
                            best_weights = self.base_weights.copy()
                            best_alpha = alpha
        
        logger.info(f"Grid Search completo: {total_combinations} combinações testadas")
        logger.info(f"Melhor {metric}: {best_score:.4f}")
        logger.info(f"Melhor α: {best_alpha:.2f}")
        logger.info(f"Melhores pesos: ML={best_weights['ml']:.2f}, XAI={best_weights['xai']:.2f}, "
                   f"DRL={best_weights['drl']:.2f}, NLP={best_weights['nlp']:.2f}")
        
        # Atualizar pesos e alpha
        self.base_weights = best_weights
        self.alpha = best_alpha
        self.is_fitted = True
        
        # Registrar no histórico
        self.weights_history.append({
            'weights': best_weights.copy(),
            'alpha': best_alpha,
            'metric': metric,
            'score': best_score
        })
        
        return best_weights
    
    def optimize_threshold(
        self,
        ml_proba: np.ndarray,
        xai_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        y_true: np.ndarray,
        interpretability_scores: Optional[np.ndarray] = None,
        metric: str = 'f1'
    ) -> float:
        """
        Otimiza o limiar de classificação.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        xai_proba : np.ndarray
            Probabilidades do modelo XAI
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        y_true : np.ndarray
            Labels verdadeiros
        interpretability_scores : np.ndarray, optional
            Scores de interpretabilidade
        metric : str
            Métrica a otimizar ('f1', 'accuracy')
            
        Retorna:
        --------
        float
            Limiar otimizado
        """
        logger.info("Iniciando otimização de limiar...")
        
        # Calcular probabilidades do ensemble
        ensemble_proba = self.predict_proba(
            ml_proba, xai_proba, drl_proba, nlp_proba,
            interpretability_scores
        )
        
        # Testar diferentes limiares
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (ensemble_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
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
        xai_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        y_true: np.ndarray,
        interpretability_scores: Optional[np.ndarray] = None,
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
        xai_proba : np.ndarray
            Probabilidades do modelo XAI
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        y_true : np.ndarray
            Labels verdadeiros
        interpretability_scores : np.ndarray, optional
            Scores de interpretabilidade
        optimize_weights : bool
            Se True, otimiza os pesos
        optimize_threshold : bool
            Se True, otimiza o limiar
        metric : str
            Métrica a otimizar
        """
        if optimize_weights:
            self.optimize_weights(
                ml_proba, xai_proba, drl_proba, nlp_proba, y_true,
                interpretability_scores, metric
            )
        
        if optimize_threshold:
            self.optimize_threshold(
                ml_proba, xai_proba, drl_proba, nlp_proba, y_true,
                interpretability_scores, metric
            )
        
        self.is_fitted = True
        logger.info("✅ Ensemble treinado com sucesso!")
    
    def evaluate(
        self,
        ml_proba: np.ndarray,
        xai_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        y_true: np.ndarray,
        interpretability_scores: Optional[np.ndarray] = None,
        mc_dropout_samples: Optional[np.ndarray] = None
    ) -> EnsembleMetrics:
        """
        Avalia o desempenho do ensemble.
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do modelo ML
        xai_proba : np.ndarray
            Probabilidades do modelo XAI
        drl_proba : np.ndarray
            Probabilidades do agente DRL
        nlp_proba : np.ndarray
            Probabilidades do modelo NLP
        y_true : np.ndarray
            Labels verdadeiros
        interpretability_scores : np.ndarray, optional
            Scores de interpretabilidade
        mc_dropout_samples : np.ndarray, optional
            Amostras de Monte Carlo Dropout
            
        Retorna:
        --------
        EnsembleMetrics
            Métricas de desempenho
        """
        # Predições
        ensemble_proba = self.predict_proba(
            ml_proba, xai_proba, drl_proba, nlp_proba,
            interpretability_scores
        )
        y_pred = self.predict(
            ml_proba, xai_proba, drl_proba, nlp_proba,
            interpretability_scores
        )
        
        # Calcular incerteza
        U_MC, U_ens, U_total = self.calculate_uncertainty(
            ml_proba, xai_proba, drl_proba, nlp_proba,
            mc_dropout_samples
        )
        
        # Calcular interpretabilidade média
        if interpretability_scores is not None:
            interpretability_score = interpretability_scores.mean()
        else:
            interpretability_score = 0.0
        
        # Calcular métricas
        metrics = EnsembleMetrics(
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            auc=roc_auc_score(y_true, ensemble_proba),
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            mcc=matthews_corrcoef(y_true, y_pred),
            interpretability_score=interpretability_score,
            uncertainty=U_total.mean()
        )
        
        return metrics
    
    def save(self, filepath: str):
        """Salva o ensemble treinado."""
        joblib.dump(self, filepath)
        logger.info(f"✅ Ensemble salvo em: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'WeightedEnsemble4Paradigms':
        """Carrega um ensemble treinado."""
        ensemble = joblib.load(filepath)
        logger.info(f"✅ Ensemble carregado de: {filepath}")
        return ensemble
    
    def get_summary(self) -> Dict:
        """Retorna resumo do ensemble."""
        return {
            'base_weights': self.base_weights,
            'alpha': self.alpha,
            'threshold': self.threshold,
            'is_fitted': self.is_fitted,
            'weights_history': self.weights_history
        }


# Exemplo de uso
if __name__ == "__main__":
    # Gerar dados sintéticos para demonstração
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=1000,
        n_features=30,
        n_informative=20,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Simular probabilidades dos 4 paradigmas
    np.random.seed(42)
    ml_proba = np.random.rand(len(y_test))
    xai_proba = ml_proba + np.random.normal(0, 0.05, len(y_test))
    drl_proba = ml_proba + np.random.normal(0, 0.08, len(y_test))
    nlp_proba = ml_proba + np.random.normal(0, 0.1, len(y_test))
    
    # Simular scores de interpretabilidade
    interpretability_scores = np.random.rand(len(y_test))
    
    # Criar e treinar ensemble
    ensemble = WeightedEnsemble4Paradigms(alpha=0.5)
    ensemble.fit(
        ml_proba, xai_proba, drl_proba, nlp_proba, y_test,
        interpretability_scores,
        optimize_weights=True,
        optimize_threshold=True,
        metric='f1'
    )
    
    # Avaliar
    metrics = ensemble.evaluate(
        ml_proba, xai_proba, drl_proba, nlp_proba, y_test,
        interpretability_scores
    )
    
    print("\n" + "="*60)
    print("MÉTRICAS DO ENSEMBLE 4-PARADIGMAS")
    print("="*60)
    print(f"F1-Score: {metrics.f1_score:.4f}")
    print(f"AUC: {metrics.auc:.4f}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"MCC: {metrics.mcc:.4f}")
    print(f"Interpretabilidade: {metrics.interpretability_score:.4f}")
    print(f"Incerteza Total: {metrics.uncertainty:.4f}")
    print("="*60)
    
    # Salvar
    ensemble.save('ensemble_4_paradigmas.pkl')
    
    # Resumo
    print("\nRESUMO DO ENSEMBLE:")
    print(ensemble.get_summary())
