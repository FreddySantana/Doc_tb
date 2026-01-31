"""
Módulo: Xai
Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose
Data de Criação: 2024-06-01
Última Modificação: 2025-01-20
"""
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer'
]
