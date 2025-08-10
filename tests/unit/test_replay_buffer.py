"""
Unit tests for the replay buffer.
"""

import pytest
import numpy as np
import torch
from utils.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    """Test the ReplayBuffer class."""

    def test_replay_buffer_initialization(self):
        """Test that the replay buffer initializes correctly."""
        # Create replay buffer
        buffer = ReplayBuffer(capacity=100, state_dim=8, device="cpu")
        
        # Check that the buffer has the correct attributes
        assert buffer.capacity == 100
        assert buffer.state_dim == 8
        assert str(buffer.device) == "cpu"
        assert len(buffer) == 0
        
        # Check that the buffer is empty
        assert len(buffer.buffer) == 0
        
        # Check that the buffer is not ready for sampling
        assert not buffer.is_ready(batch_size=32)
    
    def test_replay_buffer_push(self):
        """Test that the replay buffer can push transitions."""
        # Create replay buffer
        buffer = ReplayBuffer(capacity=100, state_dim=8, device="cpu")
        
        # Push a transition
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        action = 1
        reward = 0.5
        next_state = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        
        # Check that the buffer has one transition
        assert len(buffer) == 1
        assert len(buffer.buffer) == 1
        
        # Check that the buffer is still not ready for sampling
        assert not buffer.is_ready(batch_size=32)
    
    def test_replay_buffer_sample(self):
        """Test that the replay buffer can sample transitions."""
        # Create replay buffer
        buffer = ReplayBuffer(capacity=100, state_dim=8, device="cpu")
        
        # Push enough transitions to sample
        for i in range(32):
            state = np.array([i/100.0] * 8, dtype=np.float32)
            action = i % 3
            reward = i / 10.0
            next_state = np.array([(i+1)/100.0] * 8, dtype=np.float32)
            done = (i % 5 == 0)
            
            buffer.push(state, action, reward, next_state, done)
        
        # Check that the buffer is ready for sampling
        assert buffer.is_ready(batch_size=16)
        
        # Sample from the buffer
        batch_size = 16
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        # Check that the samples have the correct shape
        assert states.shape == (batch_size, 8)
        assert actions.shape == (batch_size,)  # Actions are a 1D tensor
        assert rewards.shape == (batch_size,)  # Rewards are a 1D tensor
        assert next_states.shape == (batch_size, 8)
        assert dones.shape == (batch_size,)  # Dones are a 1D tensor
        
        # Check that the samples are tensors
        assert isinstance(states, torch.Tensor)
        assert isinstance(actions, torch.Tensor)
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(next_states, torch.Tensor)
        assert isinstance(dones, torch.Tensor)
    
    def test_replay_buffer_overflow(self):
        """Test that the replay buffer handles overflow correctly."""
        # Create replay buffer with small capacity
        capacity = 10
        buffer = ReplayBuffer(capacity=capacity, state_dim=8, device="cpu")
        
        # Push more transitions than the capacity
        for i in range(capacity * 2):
            state = np.array([i/100.0] * 8, dtype=np.float32)
            action = i % 3
            reward = i / 10.0
            next_state = np.array([(i+1)/100.0] * 8, dtype=np.float32)
            done = (i % 5 == 0)
            
            buffer.push(state, action, reward, next_state, done)
        
        # Check that the buffer has only capacity transitions
        assert len(buffer) == capacity
        assert len(buffer.buffer) == capacity
    
    def test_replay_buffer_clear(self):
        """Test that the replay buffer can be cleared."""
        # Create replay buffer
        buffer = ReplayBuffer(capacity=100, state_dim=8, device="cpu")
        
        # Push some transitions
        for i in range(10):
            state = np.array([i/100.0] * 8, dtype=np.float32)
            action = i % 3
            reward = i / 10.0
            next_state = np.array([(i+1)/100.0] * 8, dtype=np.float32)
            done = (i % 5 == 0)
            
            buffer.push(state, action, reward, next_state, done)
        
        # Check that the buffer has transitions
        assert len(buffer) == 10
        
        # Clear the buffer
        buffer.clear()
        
        # Check that the buffer is empty
        assert len(buffer) == 0
        assert len(buffer.buffer) == 0
    
    def test_replay_buffer_get_info(self):
        """Test that the replay buffer can return info."""
        # Create replay buffer
        buffer = ReplayBuffer(capacity=100, state_dim=8, device="cpu")
        
        # Push some transitions
        for i in range(10):
            state = np.array([i/100.0] * 8, dtype=np.float32)
            action = i % 3
            reward = i / 10.0
            next_state = np.array([(i+1)/100.0] * 8, dtype=np.float32)
            done = (i % 5 == 0)
            
            buffer.push(state, action, reward, next_state, done)
        
        # Get info
        info = buffer.get_info()
        
        # Check that the info has the correct keys
        assert 'type' in info
        assert 'capacity' in info
        assert 'current_size' in info
        assert 'state_dim' in info
        assert 'utilization' in info
        
        # Check that the info has the correct values
        assert info['type'] == 'ReplayBuffer'
        assert info['capacity'] == 100
        assert info['current_size'] == 10
        assert info['state_dim'] == 8
        assert 0 <= info['utilization'] <= 1