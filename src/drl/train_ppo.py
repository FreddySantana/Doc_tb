"""
Treinamento de Agente PPO (Proximal Policy Optimization)

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-06-15
Última Modificação: 2025-11-20

Descrição:
    Implementação do algoritmo PPO conforme Algoritmo 5 (página 127) da tese.
    
    PPO é um algoritmo de policy gradient que usa clipped surrogate objective
    para garantir atualizações de política estáveis.
    
    Características:
    - Actor-Critic com policy e value networks
    - Clipped surrogate objective (Equação 66 da tese)
    - Entropy regularization
    - Generalized Advantage Estimation (GAE)
    - Mini-batch updates

Licença: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPONetwork(nn.Module):
    """
    Rede Neural para PPO (Actor-Critic).
    
    Combina policy network e value network com pesos compartilhados.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Inicializa a rede PPO.
        
        Parâmetros:
        -----------
        state_dim : int
            Dimensão do espaço de estados
        action_dim : int
            Dimensão do espaço de ações
        hidden_dim : int
            Dimensão das camadas ocultas
        """
        super(PPONetwork, self).__init__()
        
        # Camadas compartilhadas
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy head (ator)
        self.policy = nn.Linear(hidden_dim, action_dim)
        
        # Value head (crítico)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Log std para ações contínuas (se necessário)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass retornando policy logits e value.
        
        Parâmetros:
        -----------
        state : torch.Tensor
            Estado de entrada
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (policy_logits, value)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        policy_logits = self.policy(x)
        value = self.value(x)
        
        return policy_logits, value


class PPOAgent:
    """
    Agente PPO para otimização de políticas de intervenção.
    
    Implementa o Algoritmo 5 da tese com clipped surrogate objective.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        epochs: int = 10,
        batch_size: int = 64
    ):
        """
        Inicializa o agente PPO.
        
        Parâmetros:
        -----------
        state_dim : int
            Dimensão do espaço de estados
        action_dim : int
            Dimensão do espaço de ações
        hidden_dim : int
            Dimensão das camadas ocultas
        learning_rate : float
            Taxa de aprendizado
        gamma : float
            Fator de desconto
        gae_lambda : float
            Lambda para GAE
        clip_ratio : float
            Razão de clipping (ε em Equação 66)
        entropy_coeff : float
            Coeficiente de regularização de entropia
        value_coeff : float
            Coeficiente da função de valor
        epochs : int
            Número de épocas de treinamento por batch
        batch_size : int
            Tamanho do mini-batch
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Rede neural
        self.network = PPONetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Buffer de experiências
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.training_history = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
        
        logger.info("✅ PPOAgent inicializado")
        logger.info(f"   State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"   Clip ratio: {clip_ratio}, Entropy coeff: {entropy_coeff}")
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Seleciona uma ação usando a política.
        
        Parâmetros:
        -----------
        state : np.ndarray
            Estado atual
            
        Retorna:
        --------
        Tuple[int, float, float]
            (ação, log_prob, valor)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
        
        # Distribuição de probabilidade
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Amostra ação
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        Armazena uma transição no buffer.
        
        Parâmetros:
        -----------
        state : np.ndarray
            Estado
        action : int
            Ação
        reward : float
            Recompensa
        value : float
            Valor estimado
        log_prob : float
            Log probabilidade da ação
        done : bool
            Se episódio terminou
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula Generalized Advantage Estimation (GAE).
        
        Parâmetros:
        -----------
        next_value : float
            Valor do próximo estado (para bootstrap)
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            (advantages, returns)
        """
        advantages = []
        advantage = 0.0
        
        values = self.values + [next_value]
        
        # Calcular advantages de trás para frente
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            td_error = self.rewards[t] + self.gamma * next_value - values[t]
            advantage = td_error + self.gamma * self.gae_lambda * advantage
            advantages.insert(0, advantage)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(self.values)
        
        # Normalizar advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, next_value: float = 0.0):
        """
        Atualiza a política usando PPO com clipped surrogate objective.
        
        Implementa Equação 66 da tese:
        L_clip(θ) = -E[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
        
        Parâmetros:
        -----------
        next_value : float
            Valor do próximo estado
        """
        if len(self.states) == 0:
            return
        
        # Converter para tensores
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # Calcular advantages e returns
        advantages, returns = self.compute_gae(next_value)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Treinar por múltiplas épocas
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.epochs):
            # Mini-batch updates
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for i in range(0, len(self.states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                policy_logits, values = self.network(batch_states)
                
                # Calcular log probs da nova política
                probs = F.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Razão de importância
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective (Equação 66)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy loss (regularização)
                entropy_loss = -self.entropy_coeff * entropy
                
                # Loss total
                total_loss = policy_loss + self.value_coeff * value_loss + entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Limpar buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Registrar histórico
        self.training_history['policy_losses'].append(np.mean(policy_losses))
        self.training_history['value_losses'].append(np.mean(value_losses))
        self.training_history['entropy_losses'].append(np.mean(entropy_losses))
        
        logger.info(f"PPO Update - Policy Loss: {np.mean(policy_losses):.4f}, "
                   f"Value Loss: {np.mean(value_losses):.4f}, "
                   f"Entropy: {np.mean(entropy_losses):.4f}")
    
    def save(self, filepath: str):
        """
        Salva o agente treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho para salvar o modelo
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), filepath)
        logger.info(f"✅ Agente PPO salvo em: {filepath}")
    
    def load(self, filepath: str):
        """
        Carrega um agente treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho do modelo salvo
        """
        self.network.load_state_dict(torch.load(filepath))
        logger.info(f"✅ Agente PPO carregado de: {filepath}")


def train_ppo(
    env: Any,
    agent: PPOAgent,
    num_episodes: int = 100,
    max_steps: int = 1000
) -> Dict[str, List]:
    """
    Treina o agente PPO.
    
    Parâmetros:
    -----------
    env : Any
        Ambiente de treinamento
    agent : PPOAgent
        Agente PPO
    num_episodes : int
        Número de episódios
    max_steps : int
        Máximo de passos por episódio
        
    Retorna:
    --------
    Dict[str, List]
        Histórico de treinamento
    """
    logger.info(f"Iniciando treinamento PPO por {num_episodes} episódios...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Selecionar ação
            action, log_prob, value = agent.select_action(state)
            
            # Executar ação
            next_state, reward, done, _, _ = env.step(action)
            
            # Armazenar transição
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Calcular valor do próximo estado
        if done:
            next_value = 0.0
        else:
            _, next_value = agent.network(torch.FloatTensor(state).unsqueeze(0))
            next_value = next_value.item()
        
        # Atualizar política
        agent.update(next_value)
        
        episode_rewards.append(episode_reward)
        agent.training_history['episode_rewards'].append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                       f"Avg Reward (últimos 10): {avg_reward:.4f}")
    
    logger.info("✅ Treinamento PPO concluído!")
    
    return agent.training_history


if __name__ == "__main__":
    logger.info("PPO Module - Exemplo de uso")
