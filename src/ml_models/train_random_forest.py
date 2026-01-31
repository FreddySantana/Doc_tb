"""
Treinamento de Random Forest

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-06-10
Última Modificação: 2025-11-20

Descrição:
    Implementação de Random Forest conforme Seção 4.6.3 da tese.
    
    Algoritmo 4: Random Forest
    - Número de estimadores: 100
    - Número de features por divisão: √n
    - Critério: Gini
    - Bootstrap: True
    - OOB Score: True
    - Class weight: balanced_subsample
    
    Hiperparâmetros otimizados (Tabela 17):
    - Max depth: 20
    - Min samples split: 10
    - Min samples leaf: 20
    - Max features: sqrt
    - Bootstrap: True
    - OOB score: True
    - Class weight: balanced_subsample

Licença: MIT
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestTrainer:
    """
    Treinador de Random Forest conforme a tese.
    
    Implementa o Algoritmo 4 da tese com hiperparâmetros otimizados
    via Otimização Bayesiana.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o treinador de Random Forest.
        
        Parâmetros:
        -----------
        config : Dict, optional
            Dicionário de configurações
        """
        self.config = config or {}
        
        # Hiperparâmetros conforme Tabela 17 da tese
        self.hyperparams = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 20,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'class_weight': 'balanced_subsample',
            'n_jobs': -1,
            'random_state': 42,
            'criterion': 'gini'
        }
        
        self.model = None
        self.is_fitted = False
        self.training_history = {}
        
        logger.info("✅ RandomForestTrainer inicializado")
        logger.info(f"   Hiperparâmetros: {self.hyperparams}")
    
    def build_model(self) -> RandomForestClassifier:
        """
        Constrói o modelo Random Forest.
        
        Retorna:
        --------
        RandomForestClassifier
            Modelo Random Forest configurado
        """
        self.model = RandomForestClassifier(**self.hyperparams)
        logger.info("✅ Modelo Random Forest construído")
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Treina o modelo Random Forest.
        
        Parâmetros:
        -----------
        X_train : np.ndarray
            Features de treino
        y_train : np.ndarray
            Target de treino
        X_val : np.ndarray, optional
            Features de validação
        y_val : np.ndarray, optional
            Target de validação
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de treino
        """
        logger.info("Iniciando treinamento do Random Forest...")
        
        # Construir modelo se não existir
        if self.model is None:
            self.build_model()
        
        # Treinar
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calcular métricas de treino
        y_pred_train = self.model.predict(X_train)
        y_proba_train = self.model.predict_proba(X_train)[:, 1]
        
        train_metrics = {
            'f1_score': f1_score(y_train, y_pred_train),
            'auc': roc_auc_score(y_train, y_proba_train),
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train),
            'recall': recall_score(y_train, y_pred_train),
            'mcc': matthews_corrcoef(y_train, y_pred_train),
            'oob_score': self.model.oob_score_  # OOB score conforme Equação 39 da tese
        }
        
        logger.info(f"✅ Treinamento concluído")
        logger.info(f"   F1-Score (treino): {train_metrics['f1_score']:.4f}")
        logger.info(f"   AUC (treino): {train_metrics['auc']:.4f}")
        logger.info(f"   OOB Score: {train_metrics['oob_score']:.4f}")
        
        # Validação se dados forem fornecidos
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            y_proba_val = self.model.predict_proba(X_val)[:, 1]
            
            val_metrics = {
                'f1_score': f1_score(y_val, y_pred_val),
                'auc': roc_auc_score(y_val, y_proba_val),
                'accuracy': accuracy_score(y_val, y_pred_val),
                'precision': precision_score(y_val, y_pred_val),
                'recall': recall_score(y_val, y_pred_val),
                'mcc': matthews_corrcoef(y_val, y_pred_val)
            }
            
            logger.info(f"   F1-Score (validação): {val_metrics['f1_score']:.4f}")
            logger.info(f"   AUC (validação): {val_metrics['auc']:.4f}")
            
            self.training_history['validation'] = val_metrics
        
        self.training_history['training'] = train_metrics
        
        return train_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições com o modelo treinado.
        
        Parâmetros:
        -----------
        X : np.ndarray
            Features para predição
            
        Retorna:
        --------
        np.ndarray
            Classes preditas (0 ou 1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame train() primeiro.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades de predição.
        
        Parâmetros:
        -----------
        X : np.ndarray
            Features para predição
            
        Retorna:
        --------
        np.ndarray
            Probabilidades para cada classe
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame train() primeiro.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Retorna importância das features conforme Equação 40 da tese.
        
        Equação 40 (Importância de Features):
        Importância(j) = 1/B Σ Σ p(t)·ΔG(t)
        
        Parâmetros:
        -----------
        feature_names : list, optional
            Nomes das features
            
        Retorna:
        --------
        pd.DataFrame
            DataFrame com importância das features
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame train() primeiro.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Avalia o modelo em dados de teste.
        
        Parâmetros:
        -----------
        X_test : np.ndarray
            Features de teste
        y_test : np.ndarray
            Target de teste
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de teste
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame train() primeiro.")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }
        
        logger.info("📊 Métricas de Teste (Random Forest):")
        logger.info(f"   F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"   AUC: {metrics['auc']:.4f}")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   MCC: {metrics['mcc']:.4f}")
        
        return metrics
    
    def save(self, filepath: str):
        """
        Salva o modelo treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho para salvar o modelo
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame train() primeiro.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"✅ Modelo salvo em: {filepath}")
    
    def load(self, filepath: str):
        """
        Carrega um modelo treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho do modelo salvo
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"✅ Modelo carregado de: {filepath}")


def main():
    """Exemplo de uso do Random Forest."""
    
    # Dados de exemplo
    np.random.seed(42)
    n_samples = 1000
    n_features = 78
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Dividir em treino e teste
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Criar e treinar
    trainer = RandomForestTrainer()
    trainer.train(X_train, y_train, X_test, y_test)
    
    # Avaliar
    metrics = trainer.evaluate(X_test, y_test)
    
    # Feature importance
    importance_df = trainer.get_feature_importance()
    print("\n📊 Top 10 Features mais Importantes:")
    print(importance_df.head(10))
    
    # Salvar
    trainer.save('results/ml_models/random_forest.pkl')


if __name__ == "__main__":
    main()
