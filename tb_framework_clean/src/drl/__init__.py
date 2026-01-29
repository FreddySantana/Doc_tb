"""
Módulo de Deep Reinforcement Learning (DRL)

Implementa agentes de DRL para otimização de políticas de intervenção
no tratamento da tuberculose.
"""

from .environment import TBTreatmentEnv
from .train_dqn import DQNAgent, train_dqn
from .train_ppo import PPOAgent, train_ppo
from .train_sac import SACAgent, train_sac
from .drl_pipeline import DRLPipeline

__all__ = [
    'TBTreatmentEnv',
    'DQNAgent',
    'PPOAgent',
    'SACAgent',
    'train_dqn',
    'train_ppo',
    'train_sac',
    'DRLPipeline'
]
