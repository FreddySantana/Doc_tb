"""
Módulo para treinamento de agente DQN (Deep Q-Network)

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-06-10
Última Modificação: 2025-08-20

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
Treinamento de Agente DQN (Deep Q-Network)

Implementa o algoritmo DQN para aprendizado de políticas de intervenção
conforme descrito na Seção 4.5.1 da tese.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from src.drl.environment import TBTreatmentEnv
from src.utils import setup_logger, load_config, save_model

logger = setup_logger(__name__)


class DQNNetwork(nn.Module):
    """
    Rede Neural para DQN.
    
    Arquitetura: 3 camadas totalmente conectadas com ReLU.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Inicializa a rede.
        
        Args:
            state_dim: Dimensão do estado
            action_dim: Dimensão da ação
            hidden_dim: Dimensão das camadas ocultas
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Buffer de experiências para DQN.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Inicializa o buffer.
        
        Args:
            capacity: Capacidade máxima do buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Adiciona uma experiência ao buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Amostra um batch de experiências"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agente DQN (Deep Q-Network).
    
    Implementa o algoritmo DQN com Experience Replay e Target Network.
    
    Attributes:
        config: Dicionário de configurações
        device: Dispositivo (CPU ou GPU)
        q_network: Rede Q principal
        target_network: Rede Q alvo
        optimizer: Otimizador
        replay_buffer: Buffer de experiências
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any]
    ):
        """
        Inicializa o agente DQN.
        
        Args:
            state_dim: Dimensão do estado
            action_dim: Dimensão da ação
            config: Dicionário de configurações
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hiperparâmetros
        self.gamma = config.get('drl', {}).get('gamma', 0.99)
        self.epsilon = config.get('drl', {}).get('epsilon_start', 1.0)
        self.epsilon_min = config.get('drl', {}).get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('drl', {}).get('epsilon_decay', 0.995)
        self.learning_rate = config.get('drl', {}).get('learning_rate', 0.001)
        self.batch_size = config.get('drl', {}).get('batch_size', 64)
        self.target_update_freq = config.get('drl', {}).get('target_update_freq', 10)
        
        # Redes neurais
        hidden_dim = config.get('drl', {}).get('hidden_dim', 256)
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Otimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        buffer_size = config.get('drl', {}).get('buffer_size', 10000)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        logger.info(f'DQNAgent inicializado no dispositivo: {self.device}')
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Seleciona uma ação usando epsilon-greedy.
        
        Args:
            state: Estado atual
            training: Se está em modo de treinamento
        
        Returns:
            Ação selecionada
        """
        if training and np.random.rand() < self.epsilon:
            # Exploração: ação aleatória
            return np.random.randint(0, self.q_network.fc3.out_features)
        else:
            # Exploitação: melhor ação segundo Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self) -> float:
        """
        Executa um passo de treinamento.
        
        Returns:
            Loss do treinamento
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Amostrar batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Converter para tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-values atuais
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Q-values alvo
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Atualiza a rede alvo"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decai epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Retorna o valor V(s) do estado.
        
        Args:
            state: Estado
        
        Returns:
            Valor do estado
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.max().item()


def train_dqn(
    env: TBTreatmentEnv,
    config: Dict[str, Any],
    n_episodes: int = 1000
) -> Tuple[DQNAgent, Dict[str, List[float]]]:
    """
    Treina um agente DQN.
    
    Args:
        env: Ambiente de treinamento
        config: Dicionário de configurações
        n_episodes: Número de episódios de treinamento
    
    Returns:
        Tuple (agente treinado, histórico de treinamento)
    """
    logger.info('='*80)
    logger.info('TREINANDO AGENTE DQN')
    logger.info('='*80)
    
    # Criar agente
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, config)
    
    # Histórico
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'epsilons': []
    }
    
    # Treinamento
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        done = False
        while not done:
            # Selecionar ação
            action = agent.select_action(state, training=True)
            
            # Executar ação
            next_state, reward, done, info = env.step(action)
            
            # Armazenar experiência
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Treinar
            loss = agent.train_step()
            episode_losses.append(loss)
            
            # Atualizar estado
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Atualizar target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # Decair epsilon
        agent.decay_epsilon()
        
        # Registrar histórico
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['losses'].append(np.mean(episode_losses) if episode_losses else 0)
        history['epsilons'].append(agent.epsilon)
        
        # Log
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(history['episode_rewards'][-100:])
            avg_loss = np.mean(history['losses'][-100:])
            logger.info(f'Episode {episode+1}/{n_episodes} | '
                       f'Avg Reward: {avg_reward:.3f} | '
                       f'Avg Loss: {avg_loss:.4f} | '
                       f'Epsilon: {agent.epsilon:.3f}')
    
    logger.info('\n✅ Treinamento DQN concluído!')
    
    return agent, history


def main():
    """Função principal para execução standalone"""
    from src.utils import load_data
    
    logger.info('Iniciando treinamento DQN')
    
    # Carregar configuração
    config = load_config()
    
    # Carregar dados
    logger.info('Carregando dados...')
    train_df = load_data('data/processed/train.csv')
    
    # Criar ambiente
    env = TBTreatmentEnv(train_df, config)
    
    # Treinar agente
    n_episodes = config.get('drl', {}).get('n_episodes', 1000)
    agent, history = train_dqn(env, config, n_episodes)
    
    # Salvar agente
    output_dir = Path('results/drl/dqn')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(agent.q_network.state_dict(), output_dir / 'dqn_model.pth')
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f'Agente e histórico salvos em {output_dir}')
    
    print('\n' + '='*80)
    print('✅ TREINAMENTO DQN CONCLUÍDO COM SUCESSO!')
    print('='*80)
    print(f'\n📁 Modelo salvo em: {output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
