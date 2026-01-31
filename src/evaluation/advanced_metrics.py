"""
Métricas Avançadas de Avaliação

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-07-01
Última Modificação: 2025-11-20

Descrição:
    Implementação de métricas avançadas conforme Seção 4.6.4 da tese.
    
    Equações implementadas:
    - Equação 87: MCC (Matthews Correlation Coefficient)
    - Equação 88: Teste de McNemar
    - Equação 89: Intervalos de Confiança Bootstrap

Licença: MIT
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from scipy import stats

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """
    Classe para cálculo de métricas avançadas de avaliação.
    """
    
    @staticmethod
    def matthews_correlation_coefficient(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calcula o Coeficiente de Correlação de Matthews (MCC).
        
        Conforme Equação 87 da tese:
        MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        
        O MCC é uma métrica balanceada que funciona bem com dados desbalanceados.
        Varia de -1 a +1, onde:
        - +1: Predição perfeita
        - 0: Predição aleatória
        - -1: Predição inversa perfeita
        
        Parâmetros:
        -----------
        y_true : np.ndarray
            Labels verdadeiros
        y_pred : np.ndarray
            Predições
            
        Retorna:
        --------
        float
            Valor do MCC
        """
        # Calcular matriz de confusão
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        # Calcular MCC
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            mcc = 0.0
        else:
            mcc = numerator / denominator
        
        return float(mcc)
    
    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray
    ) -> Dict[str, float]:
        """
        Realiza o Teste de McNemar para comparação de dois classificadores.
        
        Conforme Equação 88 da tese:
        χ² = (|b - c| - 1)² / (b + c)
        
        Onde:
        - b: Número de amostras onde modelo 1 acerta e modelo 2 erra
        - c: Número de amostras onde modelo 1 erra e modelo 2 acerta
        
        O teste de McNemar é apropriado para comparar dois modelos no mesmo
        conjunto de dados de teste.
        
        Parâmetros:
        -----------
        y_true : np.ndarray
            Labels verdadeiros
        y_pred1 : np.ndarray
            Predições do modelo 1
        y_pred2 : np.ndarray
            Predições do modelo 2
            
        Retorna:
        --------
        Dict[str, float]
            Dicionário com estatística do teste e p-value
        """
        # Calcular b e c
        b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
        c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
        
        # Calcular estatística do teste
        if (b + c) == 0:
            chi2_stat = 0.0
            p_value = 1.0
        else:
            chi2_stat = ((np.abs(b - c) - 1) ** 2) / (b + c)
            # P-value da distribuição chi-quadrado com 1 grau de liberdade
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        return {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'b': int(b),
            'c': int(c),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metric: str = 'f1',
        n_bootstrap: int = 1000,
        ci: float = 0.95
    ) -> Dict[str, float]:
        """
        Calcula Intervalos de Confiança Bootstrap para uma métrica.
        
        Conforme Equação 89 da tese:
        IC = [percentil(α/2), percentil(1 - α/2)]
        
        O Bootstrap é um método não-paramétrico que reamostra os dados
        com reposição para estimar a distribuição de uma estatística.
        
        Parâmetros:
        -----------
        y_true : np.ndarray
            Labels verdadeiros
        y_pred : np.ndarray
            Predições
        y_proba : np.ndarray, optional
            Probabilidades (necessário para algumas métricas)
        metric : str
            Métrica a calcular ('f1', 'auc', 'accuracy', 'mcc')
        n_bootstrap : int
            Número de amostras bootstrap
        ci : float
            Nível de confiança (e.g., 0.95 para 95%)
            
        Retorna:
        --------
        Dict[str, float]
            Dicionário com estimativa e intervalo de confiança
        """
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
        
        n_samples = len(y_true)
        bootstrap_scores = []
        
        # Gerar amostras bootstrap
        for _ in range(n_bootstrap):
            # Reamostragem com reposição
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calcular métrica
            if metric == 'f1':
                score = f1_score(y_true_boot, y_pred_boot, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true_boot, y_pred_boot)
            elif metric == 'mcc':
                score = AdvancedMetrics.matthews_correlation_coefficient(
                    y_true_boot, y_pred_boot
                )
            elif metric == 'auc':
                if y_proba is not None:
                    y_proba_boot = y_proba[indices]
                    score = roc_auc_score(y_true_boot, y_proba_boot)
                else:
                    raise ValueError("y_proba é necessário para métrica 'auc'")
            else:
                raise ValueError(f"Métrica desconhecida: {metric}")
            
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calcular intervalo de confiança
        alpha = 1 - ci
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_scores, lower_percentile)
        upper_ci = np.percentile(bootstrap_scores, upper_percentile)
        
        # Estimativa pontual
        point_estimate = np.mean(bootstrap_scores)
        std_error = np.std(bootstrap_scores)
        
        return {
            'point_estimate': float(point_estimate),
            'std_error': float(std_error),
            'lower_ci': float(lower_ci),
            'upper_ci': float(upper_ci),
            'ci_level': ci,
            'n_bootstrap': n_bootstrap,
            'bootstrap_mean': float(np.mean(bootstrap_scores)),
            'bootstrap_std': float(np.std(bootstrap_scores))
        }
    
    @staticmethod
    def bootstrap_confidence_intervals_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        n_bootstrap: int = 1000,
        ci: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula Intervalos de Confiança Bootstrap para múltiplas métricas.
        
        Parâmetros:
        -----------
        y_true : np.ndarray
            Labels verdadeiros
        y_pred : np.ndarray
            Predições
        y_proba : np.ndarray, optional
            Probabilidades
        n_bootstrap : int
            Número de amostras bootstrap
        ci : float
            Nível de confiança
            
        Retorna:
        --------
        Dict[str, Dict[str, float]]
            Dicionário com ICs para cada métrica
        """
        metrics_to_compute = ['f1', 'accuracy', 'mcc']
        if y_proba is not None:
            metrics_to_compute.append('auc')
        
        results = {}
        
        for metric in metrics_to_compute:
            logger.info(f"Calculando Bootstrap CI para {metric}...")
            results[metric] = AdvancedMetrics.bootstrap_confidence_interval(
                y_true, y_pred, y_proba, metric, n_bootstrap, ci
            )
        
        return results
    
    @staticmethod
    def print_advanced_metrics_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        y_pred_model2: Optional[np.ndarray] = None,
        n_bootstrap: int = 1000,
        ci: float = 0.95
    ):
        """
        Imprime um relatório completo de métricas avançadas.
        
        Parâmetros:
        -----------
        y_true : np.ndarray
            Labels verdadeiros
        y_pred : np.ndarray
            Predições do modelo 1
        y_proba : np.ndarray, optional
            Probabilidades
        y_pred_model2 : np.ndarray, optional
            Predições do modelo 2 (para teste de McNemar)
        n_bootstrap : int
            Número de amostras bootstrap
        ci : float
            Nível de confiança
        """
        print("\n" + "="*80)
        print("RELATÓRIO DE MÉTRICAS AVANÇADAS DE AVALIAÇÃO")
        print("="*80)
        
        # MCC
        mcc = AdvancedMetrics.matthews_correlation_coefficient(y_true, y_pred)
        print(f"\n📊 Matthews Correlation Coefficient (MCC):")
        print(f"   Valor: {mcc:.6f}")
        print(f"   Interpretação: ", end="")
        if mcc > 0.7:
            print("Excelente")
        elif mcc > 0.5:
            print("Bom")
        elif mcc > 0.3:
            print("Moderado")
        else:
            print("Fraco")
        
        # Teste de McNemar (se modelo 2 fornecido)
        if y_pred_model2 is not None:
            mcnemar_result = AdvancedMetrics.mcnemar_test(y_true, y_pred, y_pred_model2)
            print(f"\n📊 Teste de McNemar (Comparação de Modelos):")
            print(f"   χ² Estatística: {mcnemar_result['chi2_statistic']:.6f}")
            print(f"   P-value: {mcnemar_result['p_value']:.6f}")
            print(f"   Significante (α=0.05): {'Sim' if mcnemar_result['significant'] else 'Não'}")
            print(f"   b (Modelo 1 acerta, Modelo 2 erra): {mcnemar_result['b']}")
            print(f"   c (Modelo 1 erra, Modelo 2 acerta): {mcnemar_result['c']}")
        
        # Bootstrap CIs
        print(f"\n📊 Intervalos de Confiança Bootstrap ({ci*100:.0f}%):")
        bootstrap_cis = AdvancedMetrics.bootstrap_confidence_intervals_all_metrics(
            y_true, y_pred, y_proba, n_bootstrap, ci
        )
        
        for metric, ci_data in bootstrap_cis.items():
            print(f"\n   {metric.upper()}:")
            print(f"      Estimativa Pontual: {ci_data['point_estimate']:.6f}")
            print(f"      Erro Padrão: {ci_data['std_error']:.6f}")
            print(f"      IC [{ci*100:.0f}%]: [{ci_data['lower_ci']:.6f}, {ci_data['upper_ci']:.6f}]")
        
        print("\n" + "="*80)


def example_usage():
    """Exemplo de uso das métricas avançadas."""
    
    # Dados de exemplo
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_pred2 = np.random.randint(0, 2, n_samples)
    y_proba = np.random.rand(n_samples)
    
    # Calcular MCC
    mcc = AdvancedMetrics.matthews_correlation_coefficient(y_true, y_pred)
    print(f"MCC: {mcc:.6f}")
    
    # Teste de McNemar
    mcnemar_result = AdvancedMetrics.mcnemar_test(y_true, y_pred, y_pred2)
    print(f"\nTeste de McNemar:")
    print(f"  χ²: {mcnemar_result['chi2_statistic']:.6f}")
    print(f"  P-value: {mcnemar_result['p_value']:.6f}")
    
    # Bootstrap CIs
    bootstrap_cis = AdvancedMetrics.bootstrap_confidence_intervals_all_metrics(
        y_true, y_pred, y_proba, n_bootstrap=100
    )
    
    print(f"\nBootstrap CIs (95%):")
    for metric, ci_data in bootstrap_cis.items():
        print(f"  {metric}: [{ci_data['lower_ci']:.6f}, {ci_data['upper_ci']:.6f}]")
    
    # Relatório completo
    AdvancedMetrics.print_advanced_metrics_report(
        y_true, y_pred, y_proba, y_pred2, n_bootstrap=100
    )


if __name__ == "__main__":
    example_usage()
