"""
Treinamento de Regressão Logística (White Box Model)

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-06-10
Última Modificação: 2025-11-20

Descrição:
    Implementação de Regressão Logística como modelo interpretável (white box).
    
    Conforme mencionado na tese, modelos interpretáveis são essenciais para
    contextos de saúde, permitindo que profissionais entendam as decisões do modelo.
    
    Características:
    - Totalmente interpretável (coeficientes lineares)
    - Função de ativação: Sigmoid (Equação 1 da tese)
    - Regularização: L2 (Ridge)
    - Solver: LBFGS
    - Class weight: balanced

Licença: MIT
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix
)
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticRegressionWhiteBox:
    """
    Modelo de Regressão Logística Interpretável (White Box).
    
    Implementa um modelo completamente interpretável onde cada coeficiente
    representa o impacto direto de uma feature na predição.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o treinador de Regressão Logística.
        
        Parâmetros:
        -----------
        config : Dict, optional
            Dicionário de configurações
        """
        self.config = config or {}
        
        # Hiperparâmetros
        self.hyperparams = {
            'penalty': 'l2',  # Regularização L2
            'C': 1.0,  # Inverso da força de regularização
            'solver': 'lbfgs',  # Solver
            'max_iter': 1000,
            'class_weight': 'balanced',  # Balanceamento automático
            'random_state': 42
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history = {}
        self.feature_names = None
        
        logger.info("✅ LogisticRegressionWhiteBox inicializado")
        logger.info(f"   Hiperparâmetros: {self.hyperparams}")
    
    def build_model(self) -> LogisticRegression:
        """
        Constrói o modelo de Regressão Logística.
        
        Retorna:
        --------
        LogisticRegression
            Modelo de Regressão Logística configurado
        """
        self.model = LogisticRegression(**self.hyperparams)
        logger.info("✅ Modelo de Regressão Logística construído")
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Treina o modelo de Regressão Logística.
        
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
        feature_names : list, optional
            Nomes das features
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de treino
        """
        logger.info("Iniciando treinamento da Regressão Logística...")
        
        # Armazenar nomes das features
        self.feature_names = feature_names
        
        # Construir modelo se não existir
        if self.model is None:
            self.build_model()
        
        # Normalizar features (importante para Regressão Logística)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Treinar
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Calcular métricas de treino
        y_pred_train = self.model.predict(X_train_scaled)
        y_proba_train = self.model.predict_proba(X_train_scaled)[:, 1]
        
        train_metrics = {
            'f1_score': f1_score(y_train, y_pred_train),
            'auc': roc_auc_score(y_train, y_proba_train),
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train),
            'recall': recall_score(y_train, y_pred_train),
            'mcc': matthews_corrcoef(y_train, y_pred_train)
        }
        
        logger.info(f"✅ Treinamento concluído")
        logger.info(f"   F1-Score (treino): {train_metrics['f1_score']:.4f}")
        logger.info(f"   AUC (treino): {train_metrics['auc']:.4f}")
        
        # Validação se dados forem fornecidos
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_pred_val = self.model.predict(X_val_scaled)
            y_proba_val = self.model.predict_proba(X_val_scaled)[:, 1]
            
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Retorna os coeficientes do modelo (interpretabilidade).
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame com coeficientes e suas interpretações
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Chame train() primeiro.")
        
        coefficients = self.model.coef_[0]
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        else:
            feature_names = self.feature_names
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Adicionar interpretação
        coef_df['direction'] = coef_df['coefficient'].apply(
            lambda x: 'Aumenta risco' if x > 0 else 'Diminui risco'
        )
        
        return coef_df
    
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
        
        logger.info("📊 Métricas de Teste (Regressão Logística - White Box):")
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
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Modelo salvo em: {filepath}")
    
    def load(self, filepath: str):
        """
        Carrega um modelo treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho do modelo salvo
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        
        logger.info(f"✅ Modelo carregado de: {filepath}")


def main():
    """Exemplo de uso da Regressão Logística."""
    
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
    trainer = LogisticRegressionWhiteBox()
    trainer.train(X_train, y_train, X_test, y_test)
    
    # Avaliar
    metrics = trainer.evaluate(X_test, y_test)
    
    # Coeficientes (interpretabilidade)
    coef_df = trainer.get_coefficients()
    print("\n📊 Top 10 Coeficientes (Interpretabilidade):")
    print(coef_df.head(10))
    
    # Salvar
    trainer.save('results/ml_models/logistic_regression_white_box.pkl')


if __name__ == "__main__":
    main()
