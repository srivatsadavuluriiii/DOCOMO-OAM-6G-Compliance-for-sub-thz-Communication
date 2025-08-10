"""
Integration tests for the environment and simulator.
"""

import pytest
import numpy as np
import gymnasium as gym
from environment.oam_env import OAM_Env
from environment.stable_oam_env import StableOAM_Env
from simulator.channel_simulator import ChannelSimulator


class TestEnvironmentSimulatorIntegration:
    """Test the integration between the environment and simulator."""

    def test_env_simulator_initialization(self, base_config):
        """Test that the environment and simulator initialize correctly."""
        # Create simulator
        simulator = ChannelSimulator(base_config)
        
        # Create environment
        env = OAM_Env(base_config, simulator=simulator)
        
        # Check that the environment has the correct simulator
        assert env.simulator is simulator
        
        # Check that the environment has the correct observation and action spaces
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        
        # Check that the environment has the correct OAM mode range
        assert env.min_mode == base_config['oam']['min_mode']
        assert env.max_mode == base_config['oam']['max_mode']
    
    def test_env_simulator_reset(self, base_config):
        """Test that the environment and simulator reset correctly."""
        # Create simulator
        simulator = ChannelSimulator(base_config)
        
        # Create environment
        env = OAM_Env(base_config, simulator=simulator)
        
        # Reset the environment
        state, info = env.reset()
        
        # Check that the state has the correct shape
        assert state.shape == env.observation_space.shape
        
        # Check that the info dictionary has the correct keys
        assert 'episode_handovers' in info
        assert 'episode_reward' in info
        assert 'episode_throughput' in info
        assert 'episode_steps' in info
    
    def test_env_simulator_step(self, base_config):
        """Test that the environment and simulator step correctly."""
        # Create simulator
        simulator = ChannelSimulator(base_config)
        
        # Create environment
        env = OAM_Env(base_config, simulator=simulator)
        
        # Reset the environment
        state, info = env.reset()
        
        # Take a step
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Check that the next state has the correct shape
        assert next_state.shape == env.observation_space.shape
        
        # Check that the reward is a float
        assert isinstance(reward, float)
        
        # Check that terminated and truncated are booleans
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        # Check that the info dictionary has the correct keys
        assert 'episode_handovers' in info
        assert 'episode_reward' in info
        assert 'episode_throughput' in info
        assert 'episode_steps' in info
        assert 'handovers' in info
        assert 'throughput' in info
    
    def test_stable_env_simulator_integration(self, stable_reward_config):
        """Test the integration between the stable environment and simulator."""
        # Create simulator
        simulator = ChannelSimulator(stable_reward_config)
        
        # Create environment
        env = StableOAM_Env(stable_reward_config, simulator=simulator)
        
        # Reset the environment
        state, info = env.reset()
        
        # Take a step
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Check that the info dictionary has the correct keys
        assert 'episode_handovers' in info
        assert 'episode_reward' in info
        assert 'episode_throughput' in info
        assert 'episode_steps' in info
        assert 'handovers' in info
        assert 'throughput' in info
        assert 'smoothed_throughput' in info
        
        # Check that the smoothed throughput is a float
        assert isinstance(info['smoothed_throughput'], float)
    
    def test_simulator_run_step(self, base_config):
        """Test that the simulator run_step method works correctly."""
        # Create simulator
        simulator = ChannelSimulator(base_config)
        
        # Create a user position
        user_position = np.array([100.0, 0.0, 0.0])
        
        # Run a step for each OAM mode
        for mode in range(base_config['oam']['min_mode'], base_config['oam']['max_mode'] + 1):
            H, sinr = simulator.run_step(user_position, mode)
            
            # Check that H is a numpy array with the correct shape
            assert isinstance(H, np.ndarray)
            assert H.shape == (base_config['oam']['max_mode'], base_config['oam']['max_mode'])
            
            # Check that sinr is a float
            assert isinstance(sinr, float)
    
    def test_env_simulator_multiple_steps(self, base_config):
        """Test that the environment and simulator work correctly over multiple steps."""
        # Create simulator
        simulator = ChannelSimulator(base_config)
        
        # Create environment
        env = OAM_Env(base_config, simulator=simulator)
        
        # Reset the environment
        state, info = env.reset()
        
        # Take multiple steps
        for _ in range(10):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Check that the next state has the correct shape
            assert next_state.shape == env.observation_space.shape
            
            # Check that the reward is a float
            assert isinstance(reward, float)
            
            # If the episode is done, reset the environment
            if terminated or truncated:
                state, info = env.reset()
            else:
                state = next_state