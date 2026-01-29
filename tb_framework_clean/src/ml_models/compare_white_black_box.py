"""
Módulo para comparação entre modelos White Box e Black Box

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-05-25
Última Modificação: 2025-08-10

Descrição:
    Este módulo faz parte do framework multi-paradigma desenvolvido para predição
    de abandono de tratamento em pacientes com tuberculose. O framework integra
    técnicas de Machine Learning, Deep Reinforcement Learning, Natural Language
    Processing e Explainable AI.

Licença: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List
import json

# Importar trainers
from src.ml_models.train_logistic_regression import LogisticRegressionTrainer
from src.ml_models.train_decision_tree import DecisionTreeTrainer
from src.ml_models.train_xgboost import XGBoostTrainer
from src.ml_models.train_lightgbm import LightGBMTrainer
from src.ml_models.train_catboost import CatBoostTrainer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhiteBoxBlackBoxComparison:
    """
    Classe para comparar modelos white box e black box.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa a comparação.
        
        Parâmetros:
        -----------
        random_state : int
            Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.results = {}
        
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict]:
        """
        Treina todos os modelos e coleta métricas.
        
        Parâmetros:
        -----------
        X_train : pd.DataFrame
            Features de treino
        y_train : pd.Series
            Target de treino
        X_test : pd.DataFrame
            Features de teste
        y_test : pd.Series
            Target de teste
            
        Retorna:
        --------
        Dict[str, Dict]
            Resultados de todos os modelos
        """
        logger.info("="*60)
        logger.info("INICIANDO COMPARAÇÃO WHITE BOX vs BLACK BOX")
        logger.info("="*60)
        
        # White Box Models
        logger.info("\n### MODELOS WHITE BOX ###\n")
        
        # 1. Regressão Logística
        logger.info("1. Regressão Logística")
        lr_trainer = LogisticRegressionTrainer(self.random_state)
        lr_trainer.train(X_train, y_train)
        lr_metrics = lr_trainer.evaluate(X_test, y_test)
        self.results['Regressão Logística'] = {
            'type': 'white_box',
            'metrics': lr_metrics,
            'trainer': lr_trainer
        }
        
        # 2. Árvore de Decisão
        logger.info("\n2. Árvore de Decisão")
        dt_trainer = DecisionTreeTrainer(self.random_state)
        dt_trainer.train(X_train, y_train, max_depth=10)
        dt_metrics = dt_trainer.evaluate(X_test, y_test)
        self.results['Árvore de Decisão'] = {
            'type': 'white_box',
            'metrics': dt_metrics,
            'trainer': dt_trainer
        }
        
        # Black Box Models
        logger.info("\n### MODELOS BLACK BOX ###\n")
        
        # 3. XGBoost
        logger.info("3. XGBoost")
        xgb_trainer = XGBoostTrainer(self.random_state)
        xgb_trainer.train(X_train, y_train)
        xgb_metrics = xgb_trainer.evaluate(X_test, y_test)
        self.results['XGBoost'] = {
            'type': 'black_box',
            'metrics': xgb_metrics,
            'trainer': xgb_trainer
        }
        
        # 4. LightGBM
        logger.info("\n4. LightGBM")
        lgb_trainer = LightGBMTrainer(self.random_state)
        lgb_trainer.train(X_train, y_train)
        lgb_metrics = lgb_trainer.evaluate(X_test, y_test)
        self.results['LightGBM'] = {
            'type': 'black_box',
            'metrics': lgb_metrics,
            'trainer': lgb_trainer
        }
        
        # 5. CatBoost
        logger.info("\n5. CatBoost")
        cat_trainer = CatBoostTrainer(self.random_state)
        cat_trainer.train(X_train, y_train)
        cat_metrics = cat_trainer.evaluate(X_test, y_test)
        self.results['CatBoost'] = {
            'type': 'black_box',
            'metrics': cat_metrics,
            'trainer': cat_trainer
        }
        
        logger.info("\n" + "="*60)
        logger.info("COMPARAÇÃO CONCLUÍDA")
        logger.info("="*60)
        
        return self.results
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Cria tabela comparativa de todos os modelos.
        
        Retorna:
        --------
        pd.DataFrame
            Tabela com métricas de todos os modelos
        """
        data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            data.append({
                'Modelo': model_name,
                'Tipo': 'White Box' if result['type'] == 'white_box' else 'Black Box',
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC': metrics['auc']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def plot_comparison(self, save_path: str = None):
        """
        Plota comparação visual entre os modelos.
        
        Parâmetros:
        -----------
        save_path : str
            Caminho para salvar o gráfico
        """
        df = self.get_comparison_table()
        
        # Preparar dados
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        models = df['Modelo'].tolist()
        colors = ['#3498db' if t == 'White Box' else '#e74c3c' 
                 for t in df['Tipo'].tolist()]
        
        # Criar subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = df[metric].tolist()
            
            bars = ax.barh(models, values, color=colors)
            ax.set_xlabel(metric, fontsize=12)
            ax.set_xlim([0, 1])
            ax.grid(axis='x', alpha=0.3)
            
            # Adicionar valores nas barras
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + 0.01, i, f'{value:.3f}', 
                       va='center', fontsize=10)
        
        # Remover último subplot
        fig.delaxes(axes[5])
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='White Box'),
            Patch(facecolor='#e74c3c', label='Black Box')
        ]
        fig.legend(handles=legend_elements, loc='lower right', 
                  fontsize=12, frameon=True)
        
        plt.suptitle('Comparação: Modelos White Box vs Black Box', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {save_path}")
        
        plt.close()
    
    def plot_f1_comparison(self, save_path: str = None):
        """
        Plota comparação focada em F1-Score.
        
        Parâmetros:
        -----------
        save_path : str
            Caminho para salvar o gráfico
        """
        df = self.get_comparison_table()
        
        plt.figure(figsize=(12, 6))
        
        colors = ['#3498db' if t == 'White Box' else '#e74c3c' 
                 for t in df['Tipo'].tolist()]
        
        bars = plt.barh(df['Modelo'], df['F1-Score'], color=colors)
        
        # Adicionar valores
        for i, (bar, value) in enumerate(zip(bars, df['F1-Score'])):
            plt.text(value + 0.01, i, f'{value:.4f}', 
                    va='center', fontsize=11, fontweight='bold')
        
        plt.xlabel('F1-Score', fontsize=13, fontweight='bold')
        plt.xlim([0, 1])
        plt.grid(axis='x', alpha=0.3)
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='White Box (Interpretável)'),
            Patch(facecolor='#e74c3c', label='Black Box (Caixa-Preta)')
        ]
        plt.legend(handles=legend_elements, loc='lower right', 
                  fontsize=11, frameon=True)
        
        plt.title('Comparação de F1-Score: Trade-off Interpretabilidade vs Performance', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico F1 salvo em: {save_path}")
        
        plt.close()
    
    def generate_report(self, save_path: str = None) -> str:
        """
        Gera relatório em Markdown.
        
        Parâmetros:
        -----------
        save_path : str
            Caminho para salvar o relatório
            
        Retorna:
        --------
        str
            Relatório em Markdown
        """
        df = self.get_comparison_table()
        
        report = "# Comparação: Modelos White Box vs Black Box\n\n"
        report += "## Resumo Executivo\n\n"
        
        # Melhor modelo geral
        best_model = df.iloc[0]
        report += f"**Melhor Modelo (F1-Score):** {best_model['Modelo']} "
        report += f"({best_model['Tipo']}) - F1={best_model['F1-Score']:.4f}\n\n"
        
        # Melhor white box
        best_wb = df[df['Tipo'] == 'White Box'].iloc[0]
        report += f"**Melhor White Box:** {best_wb['Modelo']} - "
        report += f"F1={best_wb['F1-Score']:.4f}\n\n"
        
        # Melhor black box
        best_bb = df[df['Tipo'] == 'Black Box'].iloc[0]
        report += f"**Melhor Black Box:** {best_bb['Modelo']} - "
        report += f"F1={best_bb['F1-Score']:.4f}\n\n"
        
        # Gap
        gap = best_bb['F1-Score'] - best_wb['F1-Score']
        report += f"**Gap de Performance:** {gap:.4f} "
        report += f"({gap/best_wb['F1-Score']*100:.2f}%)\n\n"
        
        report += "## Tabela Comparativa\n\n"
        report += df.to_markdown(index=False, floatfmt=".4f")
        report += "\n\n"
        
        report += "## Análise\n\n"
        report += "### Trade-off Interpretabilidade vs Performance\n\n"
        report += "Os modelos **white box** (Regressão Logística e Árvore de Decisão) "
        report += "oferecem **máxima interpretabilidade**, permitindo que profissionais "
        report += "de saúde entendam exatamente quais fatores contribuem para a predição.\n\n"
        
        report += "Os modelos **black box** (XGBoost, LightGBM, CatBoost) apresentam "
        report += f"**melhor performance** (+{gap:.4f} F1-Score), mas são menos "
        report += "interpretáveis diretamente.\n\n"
        
        report += "### Recomendação\n\n"
        report += "Para o contexto clínico de tuberculose, recomenda-se:\n\n"
        report += "1. **Usar modelos black box** para predições (maior acurácia)\n"
        report += "2. **Aplicar XAI (SHAP/LIME)** para explicar as predições\n"
        report += "3. **Validar com modelos white box** para garantir coerência clínica\n\n"
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Relatório salvo em: {save_path}")
        
        return report
    
    def save_results(self, filepath: str):
        """
        Salva resultados em JSON.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho para salvar os resultados
        """
        # Preparar dados para JSON
        results_json = {}
        for model_name, result in self.results.items():
            results_json[model_name] = {
                'type': result['type'],
                'metrics': result['metrics']
            }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados salvos em: {filepath}")


def main():
    """Exemplo de uso."""
    # Dados de exemplo
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Comparação
    comparison = WhiteBoxBlackBoxComparison()
    comparison.train_all_models(X_train, y_train, X_test, y_test)
    
    # Tabela
    print("\n" + "="*80)
    print(comparison.get_comparison_table().to_string(index=False))
    print("="*80)
    
    # Visualizações
    comparison.plot_comparison(save_path='results/comparison/all_metrics.png')
    comparison.plot_f1_comparison(save_path='results/comparison/f1_comparison.png')
    
    # Relatório
    report = comparison.generate_report(save_path='results/comparison/report.md')
    print("\n" + report)
    
    # Salvar resultados
    comparison.save_results('results/comparison/results.json')


if __name__ == "__main__":
    main()
