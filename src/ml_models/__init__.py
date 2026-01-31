#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo ML Models
Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose
Data de Criação: 2024-07-01
Última Modificação: 2025-01-20
"""
# Imports opcionais - só importa se existir
__all__ = []

try:
    from .train_logistic_regression import LogisticRegressionTrainer
    __all__.append('LogisticRegressionTrainer')
except ImportError:
    pass

try:
    from .train_logistic_regression_white_box import LogisticRegressionWhiteBox
    __all__.append('LogisticRegressionWhiteBox')
except ImportError:
    pass

try:
    from .train_decision_tree import DecisionTreeTrainer
    __all__.append('DecisionTreeTrainer')
except ImportError:
    pass

try:
    from .train_decision_tree_white_box import DecisionTreeWhiteBox
    __all__.append('DecisionTreeWhiteBox')
except ImportError:
    pass

try:
    from .train_random_forest import RandomForestTrainer
    __all__.append('RandomForestTrainer')
except ImportError:
    pass

try:
    from .train_xgboost import XGBoostTrainer
    __all__.append('XGBoostTrainer')
except ImportError:
    pass

try:
    from .train_lightgbm import LightGBMTrainer
    __all__.append('LightGBMTrainer')
except ImportError:
    pass

try:
    from .train_catboost import CatBoostTrainer
    __all__.append('CatBoostTrainer')
except ImportError:
    pass

try:
    from .ml_pipeline import MLPipeline
    __all__.append('MLPipeline')
except ImportError:
    pass

try:
    from .compare_white_black_box import WhiteBoxBlackBoxComparison
    __all__.append('WhiteBoxBlackBoxComparison')
except ImportError:
    pass
