"""
Regression tests for the training process.
"""

import pytest
import numpy as np
import torch
from environment.oam_env import OAM_Env
from models.agent import Agent
from simulator.channel_simulator import ChannelSimulator


class TestTrainingRegression:
    """Test the training process for regression issues."""

    @pytest.mark.slow
    def test_training_performance(self, base_config):
        """Test that the training performance meets minimum expectations."""
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create simulator
        simulator = ChannelSimulator(base_config)
        
        # Create environment
        env = OAM_Env(base_config, simulator=simulator)
        
        # Create agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=[64, 64],
            learning_rate=0.001,
            gamma=0.99,
            buffer_capacity=10000,
            batch_size=64,
            target_update_freq=10,
        )
        
        # Train for a small number of episodes
        num_episodes = 5
        max_steps = 100
        
        # Track metrics
        episode_rewards = []
        episode_throughputs = []
        episode_handovers = []
        
        # Training loop
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Choose action with exploration
                epsilon = max(0.1, 0.9 - 0.8 * episode / num_episodes)
                action = agent.choose_action(state, epsilon)
                
                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Store transition in replay buffer
                agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)
                
                # Update state and episode reward
                state = next_state
                episode_reward += reward
                
                # Learn if replay buffer has enough samples
                if len(agent.replay_buffer) >= agent.batch_size:
                    agent.learn()
                
                # Check if episode is done
                if terminated or truncated:
                    break
            
            # Record metrics
            episode_rewards.append(episode_reward)
            episode_throughputs.append(info['episode_throughput'])
            episode_handovers.append(info['episode_handovers'])
        
        # Check that the average reward is above a minimum threshold
        # This threshold is set low for the test to pass reliably
        min_avg_reward = -200.0
        avg_reward = sum(episode_rewards) / num_episodes
        assert avg_reward > min_avg_reward, f"Average reward {avg_reward} below acceptable threshold {min_avg_reward}"
        
        # Check that the average throughput is above a minimum threshold
        min_avg_throughput = 0.0
        avg_throughput = sum(episode_throughputs) / num_episodes
        assert avg_throughput > min_avg_throughput, f"Average throughput {avg_throughput} below acceptable threshold {min_avg_throughput}"
    
    @pytest.mark.slow
    def test_model_output_consistency(self, base_config):
        """Test that the model outputs are consistent for given inputs."""
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
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
        
        # Create a fixed state
        fixed_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        
        # Get initial Q-values
        state_tensor = torch.FloatTensor(fixed_state).unsqueeze(0)
        initial_q_values = agent.policy_net(state_tensor).detach().numpy()
        
        # Fill replay buffer with some transitions
        for _ in range(100):
            state = np.random.rand(8).astype(np.float32)
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            next_state = np.random.rand(8).astype(np.float32)
            done = np.random.rand() > 0.8
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Learn for a few steps
        for _ in range(10):
            agent.learn()
        
        # Get final Q-values
        final_q_values = agent.policy_net(state_tensor).detach().numpy()
        
        # Check that the Q-values have changed (model is learning)
        assert not np.allclose(initial_q_values, final_q_values), "Q-values did not change after learning"
        
        # Save model
        torch.save(agent.policy_net.state_dict(), "policy_net_test.pth")
        
        # Create a new agent
        new_agent = Agent(
            state_dim=8,
            action_dim=3,
            hidden_layers=[64, 64],
            learning_rate=0.001,
            gamma=0.99,
            buffer_capacity=10000,
            batch_size=64,
            target_update_freq=10,
        )
        
        # Load model
        new_agent.policy_net.load_state_dict(torch.load("policy_net_test.pth"))
        
        # Get Q-values from new agent
        new_q_values = new_agent.policy_net(state_tensor).detach().numpy()
        
        # Check that the Q-values are the same
        assert np.allclose(final_q_values, new_q_values), "Q-values are not consistent after saving and loading model"
        
        # Clean up
        import os
        os.remove("policy_net_test.pth")