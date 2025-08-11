#!/usr/bin/env python3
"""
Replay buffer interface for dependency injection.

This module defines the interface that any replay buffer implementation
must follow, enabling dependency injection and easier testing.
"""

from typing import Protocol, Tuple, Optional, runtime_checkable
import numpy as np
import torch


@runtime_checkable
class ReplayBufferInterface(Protocol):
    """
    Interface for replay buffer implementations.
    
    This defines the contract that any replay buffer must implement,
    enabling dependency injection and easier testing.
    """
    
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
        ...
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        ...
    
    def __len__(self) -> int:
        """
        Return the current size of the buffer.
        
        Returns:
            Number of transitions in the buffer
        """
        ...
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for a batch.
        
        Args:
            batch_size: Required batch size
            
        Returns:
            True if buffer has enough samples
        """
        ...
    
    def clear(self) -> None:
        """
        Clear all transitions from the buffer.
        """
        ...
    
    def get_info(self) -> dict:
        """
        Get information about the buffer.
        
        Returns:
            Dictionary with buffer information
        """
        ... 
