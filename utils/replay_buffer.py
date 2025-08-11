import numpy as np
import torch
from collections import deque
import random
from typing import Tuple, List, Dict, Any, Optional, TYPE_CHECKING

                                                                           
if TYPE_CHECKING:
    from models.replay_buffer_interface import ReplayBufferInterface


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Stores (state, action, reward, next_state, done) tuples for RL training.
    Implements the ReplayBufferInterface for dependency injection.
    """
    
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            device: PyTorch device to use for tensor operations
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
                               
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
                                     
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
                                  
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self) >= batch_size
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer.clear()
    
    def get_info(self) -> dict:
        """
        Get information about the buffer.
        
        Returns:
            Dictionary with buffer information
        """
        return {
            'capacity': self.capacity,
            'current_size': len(self),
            'state_dim': self.state_dim,
            'device': str(self.device),
            'utilization': len(self) / self.capacity if self.capacity > 0 else 0.0,
            'type': 'ReplayBuffer'
        } 
