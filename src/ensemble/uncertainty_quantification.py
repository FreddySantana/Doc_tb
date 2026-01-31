"""
Módulo de Quantificação de Incerteza para o Ensemble

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-08-15
Última Modificação: 2025-11-20

Descrição:
    Este módulo implementa a quantificação de incerteza conforme as Equações 82-84 da tese.
    
    Equação 82 (Monte Carlo Dropout):
    Û_MC(x) = √(1/T Σ(ŷ_t(x) - ŷ_MC(x))²)
    
    Equação 83 (Variância do Ensemble):
    U_ens(x) = √(1/4 Σ(ŷ_i(x) - ŷ_ensemble(x))²)
    
    Equação 84 (Incerteza Total):
    U(x) = 0.6 · U_MC(x) + 0.4 · U_ens(x)

Licença: MIT
"""

import numpy as np
from typing import Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyQuantification:
    """
    Classe para quantificação de incerteza no ensemble.
    
    Combina duas fontes de incerteza:
    1. Incerteza epistêmica (Monte Carlo Dropout) - U_MC
    2. Variância do ensemble (entre os 4 paradigmas) - U_ens
    
    A incerteza total é uma combinação ponderada dessas duas fontes.
    """
    
    def __init__(self, mc_weight: float = 0.6, ens_weight: float = 0.4):
        """
        Inicializa o módulo de quantificação de incerteza.
        
        Parâmetros:
        -----------
        mc_weight : float
            Peso da incerteza Monte Carlo (padrão: 0.6)
        ens_weight : float
            Peso da variância do ensemble (padrão: 0.4)
        """
        assert abs(mc_weight + ens_weight - 1.0) < 1e-6, \
            f"Pesos devem somar 1.0, mas somam {mc_weight + ens_weight}"
        
        self.mc_weight = mc_weight
        self.ens_weight = ens_weight
        
        logger.info(f"✅ Quantificação de Incerteza inicializada")
        logger.info(f"   - Peso Monte Carlo: {mc_weight}")
        logger.info(f"   - Peso Ensemble: {ens_weight}")
    
    def calculate_mc_dropout_uncertainty(
        self,
        mc_samples: np.ndarray
    ) -> np.ndarray:
        """
        Calcula a incerteza epistêmica usando Monte Carlo Dropout.
        
        Conforme Equação 82 da tese:
        Û_MC(x) = √(1/T Σ(ŷ_t(x) - ŷ_MC(x))²)
        
        Parâmetros:
        -----------
        mc_samples : np.ndarray
            Amostras de Monte Carlo Dropout
            Shape: (n_samples, T) onde T é o número de amostras
            
        Retorna:
        --------
        np.ndarray
            Incerteza Monte Carlo para cada amostra
            Shape: (n_samples,)
        """
        # Validar entrada
        assert mc_samples.ndim == 2, \
            f"mc_samples deve ter 2 dimensões, mas tem {mc_samples.ndim}"
        
        n_samples, T = mc_samples.shape
        
        # Calcular média das amostras (ŷ_MC)
        y_mc_mean = np.mean(mc_samples, axis=1)  # Shape: (n_samples,)
        
        # Calcular variância: 1/T Σ(ŷ_t - ŷ_MC)²
        variance = np.mean(
            (mc_samples - y_mc_mean[:, np.newaxis]) ** 2,
            axis=1
        )
        
        # Calcular desvio padrão (incerteza)
        u_mc = np.sqrt(variance)
        
        return u_mc
    
    def calculate_ensemble_variance(
        self,
        ml_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        ensemble_proba: np.ndarray
    ) -> np.ndarray:
        """
        Calcula a variância entre as predições dos paradigmas.
        
        Conforme Equação 83 da tese:
        U_ens(x) = √(1/4 Σ(ŷ_i(x) - ŷ_ensemble(x))²)
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do ML (shape: n_samples)
        drl_proba : np.ndarray
            Probabilidades do DRL (shape: n_samples)
        nlp_proba : np.ndarray
            Probabilidades do NLP (shape: n_samples)
        ensemble_proba : np.ndarray
            Probabilidades do ensemble (shape: n_samples)
            
        Retorna:
        --------
        np.ndarray
            Variância do ensemble para cada amostra (shape: n_samples)
        """
        # Validar dimensões
        n_samples = len(ml_proba)
        assert len(drl_proba) == n_samples, "drl_proba deve ter mesmo tamanho que ml_proba"
        assert len(nlp_proba) == n_samples, "nlp_proba deve ter mesmo tamanho que ml_proba"
        assert len(ensemble_proba) == n_samples, "ensemble_proba deve ter mesmo tamanho que ml_proba"
        
        # Stack das predições: (n_samples, 3)
        paradigm_probas = np.column_stack([ml_proba, drl_proba, nlp_proba])
        
        # Calcular variância: 1/3 Σ(ŷ_i - ŷ_ensemble)²
        # Nota: Usamos 3 paradigmas (ML, DRL, NLP), não 4 (XAI foi removido)
        variance = np.mean(
            (paradigm_probas - ensemble_proba[:, np.newaxis]) ** 2,
            axis=1
        )
        
        # Calcular desvio padrão (variância do ensemble)
        u_ens = np.sqrt(variance)
        
        return u_ens
    
    def calculate_total_uncertainty(
        self,
        ml_proba: np.ndarray,
        drl_proba: np.ndarray,
        nlp_proba: np.ndarray,
        ensemble_proba: np.ndarray,
        mc_samples: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula a incerteza total combinando Monte Carlo Dropout e variância do ensemble.
        
        Conforme Equação 84 da tese:
        U(x) = 0.6 · U_MC(x) + 0.4 · U_ens(x)
        
        Parâmetros:
        -----------
        ml_proba : np.ndarray
            Probabilidades do ML
        drl_proba : np.ndarray
            Probabilidades do DRL
        nlp_proba : np.ndarray
            Probabilidades do NLP
        ensemble_proba : np.ndarray
            Probabilidades do ensemble
        mc_samples : np.ndarray, optional
            Amostras de Monte Carlo Dropout
            Se None, usa apenas a variância do ensemble
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (u_total, u_mc, u_ens)
            - u_total: Incerteza total
            - u_mc: Incerteza Monte Carlo
            - u_ens: Variância do ensemble
        """
        # Calcular variância do ensemble
        u_ens = self.calculate_ensemble_variance(
            ml_proba, drl_proba, nlp_proba, ensemble_proba
        )
        
        # Calcular incerteza Monte Carlo se amostras forem fornecidas
        if mc_samples is not None:
            u_mc = self.calculate_mc_dropout_uncertainty(mc_samples)
        else:
            # Se não há amostras MC, usar apenas variância do ensemble
            logger.warning("Amostras de Monte Carlo Dropout não fornecidas. "
                          "Usando apenas variância do ensemble.")
            u_mc = np.zeros_like(u_ens)
        
        # Calcular incerteza total (Equação 84)
        u_total = self.mc_weight * u_mc + self.ens_weight * u_ens
        
        return u_total, u_mc, u_ens
    
    def get_uncertainty_metrics(
        self,
        u_total: np.ndarray,
        u_mc: np.ndarray,
        u_ens: np.ndarray
    ) -> dict:
        """
        Calcula métricas de resumo para a incerteza.
        
        Parâmetros:
        -----------
        u_total : np.ndarray
            Incerteza total
        u_mc : np.ndarray
            Incerteza Monte Carlo
        u_ens : np.ndarray
            Variância do ensemble
            
        Retorna:
        --------
        dict
            Dicionário com métricas de incerteza
        """
        metrics = {
            'u_total_mean': np.mean(u_total),
            'u_total_std': np.std(u_total),
            'u_total_min': np.min(u_total),
            'u_total_max': np.max(u_total),
            'u_mc_mean': np.mean(u_mc),
            'u_mc_std': np.std(u_mc),
            'u_ens_mean': np.mean(u_ens),
            'u_ens_std': np.std(u_ens),
            'mc_contribution': self.mc_weight,
            'ens_contribution': self.ens_weight
        }
        
        return metrics
    
    def print_uncertainty_report(self, metrics: dict):
        """
        Imprime um relatório de incerteza.
        
        Parâmetros:
        -----------
        metrics : dict
            Dicionário com métricas de incerteza
        """
        print("\n" + "="*70)
        print("RELATÓRIO DE QUANTIFICAÇÃO DE INCERTEZA")
        print("="*70)
        
        print("\n📊 Incerteza Total U(x):")
        print(f"   Média:  {metrics['u_total_mean']:.6f}")
        print(f"   Desvio: {metrics['u_total_std']:.6f}")
        print(f"   Min:    {metrics['u_total_min']:.6f}")
        print(f"   Max:    {metrics['u_total_max']:.6f}")
        
        print("\n🔴 Incerteza Monte Carlo U_MC(x):")
        print(f"   Média:  {metrics['u_mc_mean']:.6f}")
        print(f"   Desvio: {metrics['u_mc_std']:.6f}")
        print(f"   Contribuição: {metrics['mc_contribution']*100:.1f}%")
        
        print("\n🟢 Variância do Ensemble U_ens(x):")
        print(f"   Média:  {metrics['u_ens_mean']:.6f}")
        print(f"   Desvio: {metrics['u_ens_std']:.6f}")
        print(f"   Contribuição: {metrics['ens_contribution']*100:.1f}%")
        
        print("\n" + "="*70)


def example_usage():
    """Exemplo de uso do módulo de quantificação de incerteza."""
    
    # Criar instância
    uq = UncertaintyQuantification(mc_weight=0.6, ens_weight=0.4)
    
    # Dados de exemplo
    np.random.seed(42)
    n_samples = 100
    T = 50  # Número de amostras Monte Carlo
    
    # Simular probabilidades dos paradigmas
    ml_proba = np.random.rand(n_samples)
    drl_proba = np.random.rand(n_samples)
    nlp_proba = np.random.rand(n_samples)
    ensemble_proba = (0.50 * ml_proba + 0.30 * drl_proba + 0.20 * nlp_proba)
    
    # Simular amostras de Monte Carlo Dropout
    mc_samples = np.random.rand(n_samples, T)
    
    # Calcular incerteza total
    u_total, u_mc, u_ens = uq.calculate_total_uncertainty(
        ml_proba, drl_proba, nlp_proba, ensemble_proba, mc_samples
    )
    
    # Obter métricas
    metrics = uq.get_uncertainty_metrics(u_total, u_mc, u_ens)
    
    # Imprimir relatório
    uq.print_uncertainty_report(metrics)
    
    print("\n✅ Exemplo de uso concluído com sucesso!")


if __name__ == "__main__":
    example_usage()
