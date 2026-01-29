"""
Módulo para treinamento de modelo de Árvore de Decisão

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-03-18
Última Modificação: 2025-05-22

Descrição:
    Este módulo faz parte do framework multi-paradigma desenvolvido para predição
    de abandono de tratamento em pacientes com tuberculose. O framework integra
    técnicas de Machine Learning, Deep Reinforcement Learning, Natural Language
    Processing e Explainable AI.

Licença: MIT
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score
)
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionTreeTrainer:
    """
    Trainer para Árvore de Decisão (modelo interpretável).
    
    Vantagens:
    ----------
    - Máxima interpretabilidade (regras claras)
    - Não assume distribuição dos dados
    - Captura interações não-lineares
    - Fácil visualização
    
    Limitações:
    -----------
    - Tendência a overfitting
    - Instável (pequenas mudanças nos dados afetam a árvore)
    - Pode ter menor performance que modelos ensemble
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa o trainer.
        
        Parâmetros:
        -----------
        random_state : int
            Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        class_weight: str = 'balanced'
    ) -> DecisionTreeClassifier:
        """
        Treina o modelo de Árvore de Decisão.
        
        Parâmetros:
        -----------
        X_train : pd.DataFrame
            Features de treino
        y_train : pd.Series
            Target de treino
        max_depth : int
            Profundidade máxima da árvore
        min_samples_split : int
            Número mínimo de amostras para split
        min_samples_leaf : int
            Número mínimo de amostras por folha
        class_weight : str
            Balanceamento de classes ('balanced' ou None)
            
        Retorna:
        --------
        DecisionTreeClassifier
            Modelo treinado
        """
        logger.info("Iniciando treinamento de Árvore de Decisão...")
        
        # Criar modelo
        self.model = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            criterion='gini'
        )
        
        # Treinar
        self.model.fit(X_train, y_train)
        
        # Extrair importância das features
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Treinamento concluído!")
        logger.info(f"Profundidade da árvore: {self.model.get_depth()}")
        logger.info(f"Número de folhas: {self.model.get_n_leaves()}")
        logger.info(f"Número de features: {X_train.shape[1]}")
        logger.info(f"Número de amostras: {X_train.shape[0]}")
        
        return self.model
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Realiza validação cruzada.
        
        Parâmetros:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        cv : int
            Número de folds
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de validação cruzada
        """
        logger.info(f"Realizando validação cruzada ({cv} folds)...")
        
        # Validação cruzada estratificada
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Calcular métricas
        f1_scores = cross_val_score(self.model, X, y, cv=skf, scoring='f1')
        auc_scores = cross_val_score(self.model, X, y, cv=skf, scoring='roc_auc')
        
        metrics = {
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std()
        }
        
        logger.info(f"F1-Score: {metrics['f1_mean']:.4f} (+/- {metrics['f1_std']:.4f})")
        logger.info(f"AUC: {metrics['auc_mean']:.4f} (+/- {metrics['auc_std']:.4f})")
        
        return metrics
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Avalia o modelo no conjunto de teste.
        
        Parâmetros:
        -----------
        X_test : pd.DataFrame
            Features de teste
        y_test : pd.Series
            Target de teste
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de avaliação
        """
        logger.info("Avaliando modelo...")
        
        # Predições
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log
        logger.info("=== Resultados no Teste ===")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"AUC:       {metrics['auc']:.4f}")
        
        # Classification report
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retorna as features mais importantes.
        
        Parâmetros:
        -----------
        top_n : int
            Número de top features
            
        Retorna:
        --------
        pd.DataFrame
            Top features com importâncias
        """
        if self.feature_importance is None:
            raise ValueError("Modelo não treinado ainda!")
        
        return self.feature_importance.head(top_n)
    
    def get_decision_rules(self, max_depth: int = 5) -> str:
        """
        Retorna as regras de decisão em texto.
        
        Parâmetros:
        -----------
        max_depth : int
            Profundidade máxima a exibir
            
        Retorna:
        --------
        str
            Regras de decisão
        """
        if self.model is None:
            raise ValueError("Modelo não treinado ainda!")
        
        feature_names = self.feature_importance['feature'].tolist()
        rules = export_text(self.model, feature_names=feature_names, max_depth=max_depth)
        
        return rules
    
    def plot_tree_structure(
        self,
        max_depth: int = 3,
        save_path: str = None
    ):
        """
        Plota a estrutura da árvore.
        
        Parâmetros:
        -----------
        max_depth : int
            Profundidade máxima a plotar
        save_path : str
            Caminho para salvar o gráfico
        """
        feature_names = self.feature_importance['feature'].tolist()
        
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=['Não Abandono', 'Abandono'],
            filled=True,
            rounded=True,
            max_depth=max_depth,
            fontsize=10
        )
        plt.title(f'Estrutura da Árvore de Decisão (Profundidade {max_depth})')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Árvore salva em: {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: str = None
    ):
        """
        Plota a importância das features.
        
        Parâmetros:
        -----------
        top_n : int
            Número de features a plotar
        save_path : str
            Caminho para salvar o gráfico
        """
        top_features = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importância')
        plt.title(f'Top {top_n} Features - Árvore de Decisão')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: str = None
    ):
        """
        Plota a matriz de confusão.
        
        Parâmetros:
        -----------
        X_test : pd.DataFrame
            Features de teste
        y_test : pd.Series
            Target de teste
        save_path : str
            Caminho para salvar o gráfico
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão - Árvore de Decisão')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusão salva em: {save_path}")
        
        plt.close()
    
    def save_model(self, filepath: str):
        """
        Salva o modelo treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho para salvar o modelo
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carrega um modelo treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho do modelo salvo
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.random_state = model_data['random_state']
        
        logger.info(f"Modelo carregado de: {filepath}")


def main():
    """Exemplo de uso."""
    # Dados de exemplo
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinar
    trainer = DecisionTreeTrainer()
    trainer.train(X_train, y_train, max_depth=10)
    
    # Validação cruzada
    cv_metrics = trainer.cross_validate(X_train, y_train)
    
    # Avaliar
    test_metrics = trainer.evaluate(X_test, y_test)
    
    # Visualizações
    trainer.plot_tree_structure(max_depth=3, save_path='results/decision_tree/tree_structure.png')
    trainer.plot_feature_importance(save_path='results/decision_tree/feature_importance.png')
    trainer.plot_confusion_matrix(X_test, y_test, save_path='results/decision_tree/confusion_matrix.png')
    
    # Regras de decisão
    print("\n=== Regras de Decisão (Profundidade 3) ===")
    print(trainer.get_decision_rules(max_depth=3))
    
    # Salvar
    trainer.save_model('models/decision_tree/model.pkl')
    
    print("\n=== Top 10 Features ===")
    print(trainer.get_feature_importance(10))


if __name__ == "__main__":
    main()
