"""
Benchmark tests for the RL agent.
"""

import numpy as np
import pytest
import torch
from models.agent import Agent
from utils.replay_buffer import ReplayBuffer


@pytest.fixture
def agent():
    """Create an agent for testing."""
    state_dim = 8
    action_dim = 3
    hidden_layers = [64, 64]
    learning_rate = 0.001
    gamma = 0.99
    buffer_capacity = 10000
    batch_size = 32
    target_update_freq = 10
    
    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
    )
    
    # Fill the replay buffer
    for _ in range(100):
        state = np.random.rand(state_dim).astype(np.float32)
        action = np.random.randint(0, action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim).astype(np.float32)
        done = np.random.rand() > 0.8
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
    
    return agent


def test_agent_choose_action_deterministic(benchmark, agent):
    """Benchmark the choose_action method with deterministic policy."""
    state = np.random.rand(8).astype(np.float32)
    
    benchmark(agent.choose_action, state, 0.0)  # epsilon = 0.0 for deterministic


def test_agent_choose_action_stochastic(benchmark, agent):
    """Benchmark the choose_action method with stochastic policy."""
    state = np.random.rand(8).astype(np.float32)
    
    benchmark(agent.choose_action, state, 0.5)  # epsilon = 0.5 for stochastic


def test_agent_learn(benchmark, agent):
    """Benchmark the learn method."""
    benchmark(agent.learn)


def test_agent_update_target_net(benchmark, agent):
    """Benchmark the update_target_net method."""
    benchmark(agent.update_target_net)