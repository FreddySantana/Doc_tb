"""
Módulo: Utils
Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose
Data de Criação: 2024-06-01
Última Modificação: 2025-01-20
"""
import logging
import random
import numpy as np
import yaml
import pandas as pd
import joblib
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Carrega arquivo de configuração YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega dados de um arquivo CSV."""
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str):
    """Salva DataFrame em arquivo CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: str):
    """Salva modelo usando joblib."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Carrega modelo usando joblib."""
    return joblib.load(filepath)


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Configura logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def set_seed(seed: int = 42):
    """Define seed para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)


__all__ = [
    'load_config',
    'load_data',
    'save_data',
    'save_model',
    'load_model',
    'setup_logger',
    'set_seed'
]
