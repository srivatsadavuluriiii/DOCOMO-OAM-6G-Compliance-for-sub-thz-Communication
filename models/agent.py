import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple, Dict, Any, Optional
           
            

                                                              
                                                          
                               


import sys
import os

                                                                     
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

                                           
try:
    from .dqn_model import DQNModel
    from .replay_buffer_interface import ReplayBufferInterface
except ImportError:
    from models.dqn_model import DQNModel
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
        learning_rate: float = 1e-3,  # Increase learning rate to match reward scale
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: Optional[torch.device] = None,
        replay_buffer: Optional[ReplayBufferInterface] = None,
        double_dqn: bool = False,
        dueling_dqn: bool = False
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
            double_dqn (bool): Whether to use Double DQN
            dueling_dqn (bool): Whether to use Dueling DQN architecture
            
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
        self.double_dqn = double_dqn
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
                             
        # Store dimensions first (needed for replay buffer initialization)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = DQNModel(state_dim, action_dim, hidden_layers, dueling=dueling_dqn).to(self.device)
        self.target_net = DQNModel(state_dim, action_dim, hidden_layers, dueling=dueling_dqn).to(self.device)
        
                                                          
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()                                             
        
                              
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        
                                                         
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        else:
                                                               
            self.replay_buffer = ReplayBuffer(buffer_capacity, self.state_dim, self.device)
        
                             
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
                           
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
                                         
        if random.random() < epsilon:
                                  
            return random.randint(0, self.action_dim - 1)
        else:
                                  
            with torch.no_grad():
                                                                 
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                                                  
                q_values = self.policy_net(state_tensor)
                
                                                    
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
                                            
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0
        
                                               
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
                                                     
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
                                 
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Select action with policy_net, evaluate with target_net
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(1)[0]
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            target_q_values = target_q_values.unsqueeze(1)
        
                                                 
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
                                     
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping - more reasonable bounds                                                       
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
                          
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
                                                              
        self.learn_steps += 1
        if self.target_update_freq > 0 and (self.learn_steps % self.target_update_freq == 0):
            self.update_target_network()
        
                                                                  
        return float(loss_value)
    
    def update_target_network(self) -> None:
        """Update the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
                                                           
    def update_target_net(self) -> None:
        self.update_target_network()
    
    def end_episode(self) -> None:
        """
        Perform end-of-episode updates.
        
        Updates the target network if needed.
        """
        self.episode_count += 1
        
                                            
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
