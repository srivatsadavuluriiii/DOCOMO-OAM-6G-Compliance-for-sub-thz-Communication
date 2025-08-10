import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple, Dict, Any, Optional
# import os
# import sys

# # Use centralized path management instead of sys.path.append
# from utils.path_utils import ensure_project_root_in_path
# ensure_project_root_in_path()


import sys
import os

# Ensure project root is on sys.path before importing project modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

# Support both package and script execution
try:
    from .dqn_model import DQN
    from .replay_buffer_interface import ReplayBufferInterface
except ImportError:
    from models.dqn_model import DQN
    from models.replay_buffer_interface import ReplayBufferInterface
from utils.replay_buffer import ReplayBuffer


class Agent:
    """
    RL agent that makes OAM handover decisions using a DQN.
    
    This agent manages the policy network, target network, and learning process.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [128, 128],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: Optional[torch.device] = None,
        replay_buffer: Optional[ReplayBufferInterface] = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim (int): Dimension of the state space (default: 8 for OAM environment)
            action_dim (int): Number of possible actions (default: 3 for STAY/UP/DOWN)
            hidden_layers (List[int]): List of hidden layer sizes for the DQN [128, 128]
            learning_rate (float): Learning rate for the optimizer (default: 1e-4)
            gamma (float): Discount factor for future rewards (default: 0.99)
            buffer_capacity (int): Maximum capacity of the replay buffer (default: 50000)
            batch_size (int): Batch size for training (default: 64)
            target_update_freq (int): Episodes between target network updates (default: 10)
            device (Optional[torch.device]): PyTorch device to use (defaults to CUDA if available)
            replay_buffer (Optional[ReplayBufferInterface]): Custom replay buffer for dependency injection
            
        Returns:
            Agent: Initialized DQN agent instance.
            
        Example:
            >>> agent = Agent(state_dim=8, action_dim=3)  # Basic agent
            >>> agent = Agent(
            ...     state_dim=8,
            ...     action_dim=3,
            ...     hidden_layers=[256, 128, 64],
            ...     learning_rate=1e-3,
            ...     gamma=0.95,
            ...     buffer_capacity=100000,
            ...     batch_size=128
            ... )  # Custom configuration
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_layers).to(self.device)
        
        # Copy policy network parameters to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        
        # Initialize replay buffer (dependency injection)
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        else:
            # Default to standard ReplayBuffer if none provided
            self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, self.device)
        
        # Set hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Training tracking
        self.episode_count = 0
        self.loss_history = []
        self.learn_steps = 0
    
    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state vector (8-dimensional for OAM environment)
            epsilon (float): Exploration probability (0.0 = greedy, 1.0 = random). Default: 0.0.
            
        Returns:
            int: Selected action (0: STAY, 1: UP, 2: DOWN)
            
        Example:
            >>> action = agent.choose_action(state, epsilon=0.1)  # 10% exploration
            >>> action = agent.choose_action(state, epsilon=0.0)  # Greedy action
        """
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Choose random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Choose greedy action
            with torch.no_grad():
                # Convert state to tensor and add batch dimension
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get Q-values from policy network
                q_values = self.policy_net(state_tensor)
                
                # Choose action with highest Q-value
                return torch.argmax(q_values).item()
    
    def learn(self) -> Optional[float]:
        """
        Perform one learning step using experience replay.
        
        Returns:
            Optional[Dict[str, float]]: Training metrics (loss, etc.) or None if buffer too small
            
        Example:
            >>> metrics = agent.learn()
            >>> if metrics:
            ...     print(f"Training loss: {metrics['loss']:.4f}")
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Get Q-values for current states and actions
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            # Get maximum Q-value for next states from target network
            next_q_values = self.target_net(next_states).max(1)[0]
            
            # Compute target Q-values using Bellman equation
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            # Reshape to match q_values
            target_q_values = target_q_values.unsqueeze(1)
        
        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # Store loss value
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        # Increment learn steps and update target periodically
        self.learn_steps += 1
        if self.target_update_freq > 0 and (self.learn_steps % self.target_update_freq == 0):
            self.update_target_network()
        
        # Return scalar loss for backward compatibility with tests
        return float(loss_value)
    
    def update_target_network(self) -> None:
        """Update the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    # Backward-compatible alias expected by some benchmarks
    def update_target_net(self) -> None:
        self.update_target_network()
    
    def end_episode(self) -> None:
        """
        Perform end-of-episode updates.
        
        Updates the target network if needed.
        """
        self.episode_count += 1
        
        # Update target network if it's time
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def save_models(self, save_dir: str) -> None:
        """
        Save both policy and target networks.
        
        Args:
            save_dir (str): Directory to save the models
            
        Example:
            >>> agent.save_models("models/agent_final")
        """
        os.makedirs(save_dir, exist_ok=True)
        
        policy_path = os.path.join(save_dir, "policy_net.pth")
        target_path = os.path.join(save_dir, "target_net.pth")
        
        self.policy_net.save(policy_path)
        self.target_net.save(target_path)
    
    def load_models(self, save_dir: str) -> None:
        """
        Load both policy and target networks.
        
        Args:
            save_dir (str): Directory to load the models from
            
        Example:
            >>> agent.load_models("models/agent_final")
        """
        policy_path = os.path.join(save_dir, "policy_net.pth")
        target_path = os.path.join(save_dir, "target_net.pth")
        
        self.policy_net.load(policy_path, self.device)
        self.target_net.load(target_path, self.device) 