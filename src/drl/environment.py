"""
Ambiente de Deep Reinforcement Learning para predição de abandono

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-06-01
Última Modificação: 2025-08-15

Descrição:
    Este módulo faz parte do framework multi-paradigma desenvolvido para predição
    de abandono de tratamento em pacientes com tuberculose. O framework integra
    técnicas de Machine Learning, Deep Reinforcement Learning, Natural Language
    Processing e Explainable AI.

Licença: MIT
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ambiente de Deep Reinforcement Learning para Tratamento de TB

Implementa o ambiente de simulação para treinamento de agentes DRL
conforme descrito na Seção 4.5 da tese.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

from src.utils import setup_logger

logger = setup_logger(__name__)


class TBTreatmentEnv(gym.Env):
    """
    Ambiente OpenAI Gym para tratamento de tuberculose.
    
    O ambiente simula o processo de tratamento de TB ao longo de 6 meses,
    onde o agente deve decidir quais intervenções aplicar para evitar o abandono.
    
    Estado (State):
        - 78 características do paciente
        - Mês atual do tratamento (1-6)
        - Histórico de intervenções
    
    Ação (Action):
        - 12 possíveis intervenções:
          0: Nenhuma intervenção
          1: Visita domiciliar
          2: Contato telefônico
          3: TDO (Tratamento Diretamente Observado)
          4: Apoio psicossocial
          5: Suporte financeiro
          6: Educação em saúde
          7: Acompanhamento intensivo
          8: Encaminhamento para especialista
          9: Grupo de apoio
          10: Incentivos materiais
          11: Transporte facilitado
    
    Recompensa (Reward):
        - +1.0: Paciente completa o tratamento
        - -1.0: Paciente abandona o tratamento
        - -0.1: Penalidade por intervenção (custo)
    
    Attributes:
        config: Dicionário de configurações
        data: DataFrame com dados dos pacientes
        current_patient: Índice do paciente atual
        current_month: Mês atual do tratamento (1-6)
        intervention_history: Histórico de intervenções
        done: Se o episódio terminou
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        target_col: str = 'desfecho'
    ):
        """
        Inicializa o ambiente.
        
        Args:
            data: DataFrame com dados dos pacientes
            config: Dicionário de configurações
            target_col: Nome da coluna de desfecho
        """
        super(TBTreatmentEnv, self).__init__()
        
        self.config = config
        self.data = data.copy()
        self.target_col = target_col
        
        # Separar features e target
        self.X = data.drop(target_col, axis=1).values
        self.y = data[target_col].values
        
        self.n_patients = len(self.X)
        self.n_features = self.X.shape[1]
        self.n_months = 6
        self.n_actions = 12
        
        # Definir espaços de observação e ação
        # Estado: features do paciente + mês atual + histórico de intervenções
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 1 + self.n_actions,),
            dtype=np.float32
        )
        
        # Ação: 12 possíveis intervenções
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Estado interno
        self.current_patient = None
        self.current_month = None
        self.intervention_history = None
        self.done = False
        
        logger.info(f'TBTreatmentEnv inicializado com {self.n_patients} pacientes')
        logger.info(f'Espaço de observação: {self.observation_space.shape}')
        logger.info(f'Espaço de ação: {self.n_actions} ações')
    
    def reset(self) -> np.ndarray:
        """
        Reseta o ambiente para um novo episódio.
        
        Returns:
            Estado inicial
        """
        # Selecionar paciente aleatório
        self.current_patient = np.random.randint(0, self.n_patients)
        self.current_month = 1
        self.intervention_history = np.zeros(self.n_actions, dtype=np.float32)
        self.done = False
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executa uma ação no ambiente.
        
        Args:
            action: Ação a ser executada (0-11)
        
        Returns:
            Tuple (próximo_estado, recompensa, done, info)
        """
        if self.done:
            raise RuntimeError('Episódio já terminou. Chame reset().')
        
        # Registrar intervenção
        if action > 0:  # Ação 0 = nenhuma intervenção
            self.intervention_history[action] += 1
        
        # Avançar para o próximo mês
        self.current_month += 1
        
        # Calcular recompensa
        reward = 0.0
        
        # Penalidade por intervenção (custo)
        if action > 0:
            reward -= 0.1
        
        # Verificar se o tratamento terminou
        if self.current_month > self.n_months:
            self.done = True
            
            # Recompensa final baseada no desfecho real
            outcome = self.y[self.current_patient]
            
            if outcome == 0:  # Não abandonou (completou tratamento)
                reward += 1.0
            else:  # Abandonou
                reward -= 1.0
        
        # Informações adicionais
        info = {
            'patient_id': self.current_patient,
            'month': self.current_month,
            'action': action,
            'outcome': self.y[self.current_patient] if self.done else None
        }
        
        return self._get_state(), reward, self.done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Retorna o estado atual do ambiente.
        
        Returns:
            Estado como array numpy
        """
        # Features do paciente
        patient_features = self.X[self.current_patient]
        
        # Mês atual (normalizado)
        month_feature = np.array([self.current_month / self.n_months], dtype=np.float32)
        
        # Histórico de intervenções
        intervention_features = self.intervention_history.copy()
        
        # Concatenar tudo
        state = np.concatenate([
            patient_features,
            month_feature,
            intervention_features
        ]).astype(np.float32)
        
        return state
    
    def render(self, mode='human') -> None:
        """
        Renderiza o estado atual do ambiente.
        
        Args:
            mode: Modo de renderização
        """
        if mode == 'human':
            print(f'\n=== Ambiente TB Treatment ===')
            print(f'Paciente: {self.current_patient}')
            print(f'Mês: {self.current_month}/{self.n_months}')
            print(f'Intervenções aplicadas: {self.intervention_history.sum():.0f}')
            print(f'Done: {self.done}')
            if self.done:
                outcome = 'Completou' if self.y[self.current_patient] == 0 else 'Abandonou'
                print(f'Desfecho: {outcome}')
            print('='*30)
    
    def close(self) -> None:
        """Fecha o ambiente."""
        pass
    
    def seed(self, seed: Optional[int] = None) -> None:
        """
        Define a seed para reprodutibilidade.
        
        Args:
            seed: Seed para o gerador de números aleatórios
        """
        np.random.seed(seed)


def test_environment():
    """Função de teste do ambiente"""
    from src.utils import load_data, load_config
    
    logger.info('Testando TBTreatmentEnv...')
    
    # Carregar dados
    config = load_config()
    train_df = load_data('data/processed/train.csv')
    
    # Criar ambiente
    env = TBTreatmentEnv(train_df, config)
    
    # Testar reset
    state = env.reset()
    logger.info(f'Estado inicial shape: {state.shape}')
    
    # Testar alguns passos
    for i in range(6):
        action = env.action_space.sample()  # Ação aleatória
        next_state, reward, done, info = env.step(action)
        
        logger.info(f'Step {i+1}: action={action}, reward={reward:.2f}, done={done}')
        env.render()
        
        if done:
            break
    
    logger.info('✅ Teste do ambiente concluído com sucesso!')


if __name__ == '__main__':
    test_environment()
