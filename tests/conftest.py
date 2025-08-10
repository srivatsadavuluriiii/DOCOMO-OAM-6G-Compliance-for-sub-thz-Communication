"""
Configuration file for pytest.

This file contains fixtures and configuration for pytest.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from environment.oam_env import OAM_Env
from environment.stable_oam_env import StableOAM_Env
from simulator.channel_simulator import ChannelSimulator
from models.agent import Agent
from models.dqn_model import DQN
from utils.config_utils import load_config
from utils.hierarchical_config import load_hierarchical_config


@pytest.fixture
def base_config():
    """Return the base configuration."""
    return load_hierarchical_config("base_config_new")


@pytest.fixture
def rl_config():
    """Return the RL configuration."""
    return load_hierarchical_config("rl_config_new")


@pytest.fixture
def stable_reward_config():
    """Return the stable reward configuration."""
    return load_hierarchical_config("stable_reward_config_new")


@pytest.fixture
def extended_config():
    """Return the extended training configuration."""
    return load_hierarchical_config("extended_config_new")


@pytest.fixture
def simulator(base_config):
    """Return a simulator instance."""
    return ChannelSimulator(base_config)


@pytest.fixture
def oam_env(base_config, simulator):
    """Return an OAM environment instance."""
    return OAM_Env(base_config, simulator=simulator)


@pytest.fixture
def stable_oam_env(stable_reward_config, simulator):
    """Return a stable OAM environment instance."""
    return StableOAM_Env(stable_reward_config, simulator=simulator)


@pytest.fixture
def agent(rl_config, oam_env):
    """Return an agent instance."""
    state_dim = oam_env.observation_space.shape[0]
    action_dim = oam_env.action_space.n
    return Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=rl_config['network']['hidden_layers'],
        learning_rate=rl_config['training']['learning_rate'],
        gamma=rl_config['training']['gamma'],
        buffer_capacity=rl_config['rl_base']['replay_buffer']['capacity'],
        batch_size=rl_config['training']['batch_size'],
        target_update_freq=rl_config['training']['target_update_freq'],
    )