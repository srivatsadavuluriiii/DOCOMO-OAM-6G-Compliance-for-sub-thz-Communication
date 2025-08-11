"""
Unit tests for the agent.
"""

import pytest
import torch
import numpy as np
from models.agent import Agent
from utils.replay_buffer import ReplayBuffer


class TestAgent:
    """Test the Agent class."""

    def test_agent_initialization(self, rl_config):
        """Test that the agent initializes correctly."""
        # Create agent
        agent = Agent(
            state_dim=8,
            action_dim=3,
            hidden_layers=rl_config['rl_base']['network']['hidden_layers'],
            learning_rate=rl_config['training']['learning_rate'],
            gamma=rl_config['training']['gamma'],
            buffer_capacity=rl_config['rl_base']['replay_buffer']['capacity'],
            batch_size=rl_config['training']['batch_size'],
            target_update_freq=rl_config['training']['target_update_freq'],
        )
        
        # Check that the agent has the correct attributes
        assert agent.state_dim == 8
        assert agent.action_dim == 3
        assert agent.batch_size == rl_config['training']['batch_size']
        assert agent.gamma == rl_config['training']['gamma']
        assert agent.target_update_freq == rl_config['training']['target_update_freq']
        
        # Check that the agent has the correct networks
        assert agent.policy_net is not None
        assert agent.target_net is not None
        assert agent.optimizer is not None
        assert agent.replay_buffer is not None
        
        # Check that the replay buffer has the correct capacity
        assert agent.replay_buffer.capacity == rl_config['rl_base']['replay_buffer']['capacity']
    
    def test_agent_choose_action_deterministic(self):
        """Test that the agent chooses actions deterministically with epsilon=0."""
        # Create agent
        agent = Agent(
            state_dim=8,
            action_dim=3,
            hidden_layers=[64, 64],
            learning_rate=0.001,
            gamma=0.99,
            buffer_capacity=10000,
            batch_size=64,
            target_update_freq=10,
        )
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a state
        state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        
        # Choose action with epsilon=0 (deterministic)
        action1 = agent.choose_action(state, epsilon=0.0)
        action2 = agent.choose_action(state, epsilon=0.0)
        
        # Check that the actions are the same
        assert action1 == action2
    
    def test_agent_choose_action_stochastic(self):
        """Test that the agent chooses actions stochastically with epsilon=1."""
        # Create agent
        agent = Agent(
            state_dim=8,
            action_dim=3,
            hidden_layers=[64, 64],
            learning_rate=0.001,
            gamma=0.99,
            buffer_capacity=10000,
            batch_size=64,
            target_update_freq=10,
        )
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a state
        state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        
        # Choose action with epsilon=1 (stochastic)
        actions = [agent.choose_action(state, epsilon=1.0) for _ in range(30)]
        
        # Check that there are different actions
        assert len(set(actions)) > 1
    
    def test_agent_learn(self):
        """Test that the agent learns."""
        # Create agent
        agent = Agent(
            state_dim=8,
            action_dim=3,
            hidden_layers=[64, 64],
            learning_rate=0.001,
            gamma=0.99,
            buffer_capacity=10000,
            batch_size=4,  # Small batch size for testing
            target_update_freq=10,
        )
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Fill replay buffer with some transitions
        for _ in range(10):
            state = np.random.rand(8).astype(np.float32)
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            next_state = np.random.rand(8).astype(np.float32)
            done = np.random.rand() > 0.8
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Learn
        loss = agent.learn()
        
        # Check that the loss is not None
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_agent_target_update(self):
        """Test that the agent updates the target network."""
        # Create agent
        agent = Agent(
            state_dim=8,
            action_dim=3,
            hidden_layers=[64, 64],
            learning_rate=0.001,
            gamma=0.99,
            buffer_capacity=10000,
            batch_size=4,  # Small batch size for testing
            target_update_freq=2,  # Update target network every 2 steps
        )
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Fill replay buffer with some transitions
        for _ in range(10):
            state = np.random.rand(8).astype(np.float32)
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            next_state = np.random.rand(8).astype(np.float32)
            done = np.random.rand() > 0.8
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Get initial target network parameters
        initial_target_params = [param.clone() for param in agent.target_net.parameters()]
        
        # Learn once (step 1)
        agent.learn()
        
        # Check that target network parameters are still the same
        for i, param in enumerate(agent.target_net.parameters()):
            assert torch.allclose(param, initial_target_params[i])
        
        # Learn again (step 2)
        agent.learn()
        
        # Check that target network parameters have changed
        params_changed = False
        for i, param in enumerate(agent.target_net.parameters()):
            if not torch.allclose(param, initial_target_params[i]):
                params_changed = True
                break
        
        assert params_changed, "Target network parameters did not change after update"

    def test_multi_objective_reward_defaults_unchanged(self, base_config):
        """Default reward behavior remains reasonable without multi_objective config."""
        from environment.reward_calculator import RewardCalculator
        rc = RewardCalculator(base_config)
        r = rc.calculate_reward(throughput=1e9, sinr_dB=10.0, handover_occurred=False)
        assert r > 0
        r2 = rc.calculate_reward(throughput=0.0, sinr_dB=-10.0, handover_occurred=True)
        assert r2 <= 0

    def test_multi_objective_reward_enabled(self, base_config):
        from environment.reward_calculator import RewardCalculator
        cfg = dict(base_config)
        cfg.setdefault('rl_base', {}).setdefault('reward', {})['multi_objective'] = {
            'throughput': 1.0,
            'stability': 0.5,
            'energy': 0.0,
            'handover': 0.5,
        }
        rc = RewardCalculator(cfg)
        r_stable = rc.calculate_reward(throughput=1e9, sinr_dB=10.0, handover_occurred=False)
        r_handover = rc.calculate_reward(throughput=1e9, sinr_dB=10.0, handover_occurred=True)
        assert r_stable > r_handover