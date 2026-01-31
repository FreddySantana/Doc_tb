"""
Treinamento de Agente SAC (Soft Actor-Critic)

Autor: Frederico Guilherme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-06-15
Última Modificação: 2025-11-20

Descrição:
    Implementação do algoritmo SAC conforme Algoritmo 6 (página 129) da tese.
    
    SAC é um algoritmo off-policy que combina entropy regularization com
    Q-learning para aprender políticas exploratórias estáveis.
    
    Características:
    - Actor-Critic com policy e Q-networks
    - Entropy regularization com temperature adaptativa
    - Experience replay buffer
    - Target networks
    - Ações contínuas suportadas

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


class SACNetwork(nn.Module):
    """
    Rede Neural para SAC (Actor-Critic com Q-networks).
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Inicializa a rede SAC.
        
        Parâmetros:
        -----------
        state_dim : int
            Dimensão do espaço de estados
        action_dim : int
            Dimensão do espaço de ações
        hidden_dim : int
            Dimensão das camadas ocultas
        """
        super(SACNetwork, self).__init__()
        
        # Policy network (ator)
        self.policy_fc1 = nn.Linear(state_dim, hidden_dim)
        self.policy_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Q-networks (críticos)
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
        # Value network (opcional, para estabilidade)
        self.value_fc1 = nn.Linear(state_dim, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)
    
    def forward_policy(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass da política.
        
        Retorna média e log std da distribuição gaussiana.
        """
        x = F.relu(self.policy_fc1(state))
        x = F.relu(self.policy_fc2(x))
        
        mean = self.policy_mean(x)
        log_std = torch.clamp(self.policy_log_std(x), -20, 2)
        
        return mean, log_std
    
    def forward_q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass do Q-network 1."""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.q1_fc1(x))
        x = F.relu(self.q1_fc2(x))
        return self.q1_out(x)
    
    def forward_q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass do Q-network 2."""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.q2_fc1(x))
        x = F.relu(self.q2_fc2(x))
        return self.q2_out(x)
    
    def forward_value(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass da value network."""
        x = F.relu(self.value_fc1(state))
        x = F.relu(self.value_fc2(x))
        return self.value_out(x)


class ReplayBuffer:
    """
    Buffer de experiências para SAC (off-policy).
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Inicializa o buffer.
        
        Parâmetros:
        -----------
        capacity : int
            Capacidade máxima do buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Adiciona uma experiência ao buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Amostra um batch de experiências."""
        import random
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        """Retorna o tamanho do buffer."""
        return len(self.buffer)


class SACAgent:
    """
    Agente SAC para otimização de políticas de intervenção.
    
    Implementa o Algoritmo 6 da tese com entropy regularization adaptativa.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        buffer_capacity: int = 100000
    ):
        """
        Inicializa o agente SAC.
        
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
        tau : float
            Coeficiente de atualização suave (soft update)
        alpha : float
            Coeficiente de entropia
        auto_entropy : bool
            Se deve usar entropia adaptativa
        buffer_capacity : int
            Capacidade do replay buffer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        
        # Redes
        self.network = SACNetwork(state_dim, action_dim, hidden_dim)
        self.target_network = SACNetwork(state_dim, action_dim, hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Otimizadores
        self.policy_optimizer = optim.Adam(
            list(self.network.policy_fc1.parameters()) +
            list(self.network.policy_fc2.parameters()) +
            list(self.network.policy_mean.parameters()) +
            list(self.network.policy_log_std.parameters()),
            lr=learning_rate
        )
        
        self.q_optimizer = optim.Adam(
            list(self.network.q1_fc1.parameters()) +
            list(self.network.q1_fc2.parameters()) +
            list(self.network.q1_out.parameters()) +
            list(self.network.q2_fc1.parameters()) +
            list(self.network.q2_fc2.parameters()) +
            list(self.network.q2_out.parameters()),
            lr=learning_rate
        )
        
        # Buffer de experiências
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Entropia adaptativa
        if auto_entropy:
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        self.training_history = {
            'episode_rewards': [],
            'policy_losses': [],
            'q_losses': [],
            'alpha_values': []
        }
        
        logger.info("✅ SACAgent inicializado")
        logger.info(f"   State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"   Alpha: {alpha}, Auto entropy: {auto_entropy}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Seleciona uma ação usando a política.
        
        Parâmetros:
        -----------
        state : np.ndarray
            Estado atual
        deterministic : bool
            Se deve usar ação determinística (média)
            
        Retorna:
        --------
        np.ndarray
            Ação selecionada
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            mean, log_std = self.network.forward_policy(state_tensor)
        
        if deterministic:
            action = mean.cpu().numpy()[0]
        else:
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            action = action.cpu().numpy()[0]
        
        # Clamp para espaço de ações válido
        action = np.clip(action, -1, 1)
        
        return action
    
    def update(self, batch_size: int = 64):
        """
        Atualiza o agente SAC.
        
        Implementa Algoritmo 6 da tese com:
        - Q-learning com target networks
        - Policy gradient com entropia
        - Entropia adaptativa (opcional)
        
        Parâmetros:
        -----------
        batch_size : int
            Tamanho do mini-batch
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Amostra batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # ========== Atualizar Q-networks ==========
        with torch.no_grad():
            # Próxima ação
            next_mean, next_log_std = self.network.forward_policy(next_states)
            next_std = torch.exp(next_log_std)
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            
            # Target Q-values
            target_q1 = self.target_network.forward_q1(next_states, next_action)
            target_q2 = self.target_network.forward_q2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Soft Q-target com entropia
            target_q_value = rewards + (1 - dones) * self.gamma * (
                target_q - self.alpha * next_log_prob
            )
        
        # Q-loss
        q1 = self.network.forward_q1(states, actions)
        q2 = self.network.forward_q2(states, actions)
        
        q1_loss = F.mse_loss(q1, target_q_value)
        q2_loss = F.mse_loss(q2, target_q_value)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ========== Atualizar Policy ==========
        mean, log_std = self.network.forward_policy(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.rsample()
        log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        
        q1_new = self.network.forward_q1(states, action_sample)
        q2_new = self.network.forward_q2(states, action_sample)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_prob - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ========== Atualizar Entropia (opcional) ==========
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ========== Soft Update Target Networks ==========
        for target_param, param in zip(
            self.target_network.parameters(),
            self.network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        # Registrar histórico
        self.training_history['policy_losses'].append(policy_loss.item())
        self.training_history['q_losses'].append(q_loss.item())
        self.training_history['alpha_values'].append(self.alpha)
        
        logger.debug(f"SAC Update - Policy Loss: {policy_loss.item():.4f}, "
                    f"Q Loss: {q_loss.item():.4f}, Alpha: {self.alpha:.4f}")
    
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
        logger.info(f"✅ Agente SAC salvo em: {filepath}")
    
    def load(self, filepath: str):
        """
        Carrega um agente treinado.
        
        Parâmetros:
        -----------
        filepath : str
            Caminho do modelo salvo
        """
        self.network.load_state_dict(torch.load(filepath))
        self.target_network.load_state_dict(self.network.state_dict())
        logger.info(f"✅ Agente SAC carregado de: {filepath}")


def train_sac(
    env: Any,
    agent: SACAgent,
    num_episodes: int = 100,
    max_steps: int = 1000,
    update_frequency: int = 1
) -> Dict[str, List]:
    """
    Treina o agente SAC.
    
    Parâmetros:
    -----------
    env : Any
        Ambiente de treinamento
    agent : SACAgent
        Agente SAC
    num_episodes : int
        Número de episódios
    max_steps : int
        Máximo de passos por episódio
    update_frequency : int
        Frequência de atualização do agente
        
    Retorna:
    --------
    Dict[str, List]
        Histórico de treinamento
    """
    logger.info(f"Iniciando treinamento SAC por {num_episodes} episódios...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Selecionar ação (exploratória)
            action = agent.select_action(state, deterministic=False)
            
            # Executar ação
            next_state, reward, done, _, _ = env.step(action)
            
            # Armazenar no replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Atualizar agente
            if step % update_frequency == 0:
                agent.update(batch_size=64)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        agent.training_history['episode_rewards'].append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                       f"Avg Reward (últimos 10): {avg_reward:.4f}, "
                       f"Alpha: {agent.alpha:.4f}")
    
    logger.info("✅ Treinamento SAC concluído!")
    
    return agent.training_history


if __name__ == "__main__":
    logger.info("SAC Module - Exemplo de uso")
