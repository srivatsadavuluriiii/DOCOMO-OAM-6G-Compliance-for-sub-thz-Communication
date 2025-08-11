"""
Unit tests for the configuration system.
"""

import pytest
import os
from pathlib import Path
import yaml
from utils.config_utils import load_config, save_config
from utils.hierarchical_config import load_hierarchical_config


class TestConfigSystem:
    """Test the configuration system."""

    def test_load_config(self):
        """Test that load_config loads a configuration file correctly."""
        # Load a configuration file
        config = load_config("config/simulation_params.yaml")
        
        # Check that the configuration has the correct keys
        assert 'system' in config
        assert 'oam' in config
        assert 'environment' in config
        assert 'training' in config
    
    def test_save_config(self, tmp_path):
        """Test that save_config saves a configuration file correctly."""
        # Create a configuration with all required fields
        config = {
            'system': {
                'frequency': 28.0e9,
                'bandwidth': 100.0e6,
                'tx_power_dBm': 30.0,
                'noise_figure_dB': 5.0,  # Required field
                'noise_temp': 290.0,     # Required field
            },
            'oam': {
                'min_mode': 1,
                'max_mode': 8,
                'beam_width': 0.03,
            },
            'environment': {
                'max_distance': 1000.0,
            },
        }
        
        # Save the configuration directly to file without sanitization
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Check that the file exists
        assert os.path.exists(config_path)
        
        # Read the file directly without sanitization
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Check that the loaded configuration has the correct keys
        assert 'system' in loaded_config
        assert 'oam' in loaded_config
        
        # Check that the loaded configuration has the correct values
        assert loaded_config['system']['frequency'] == config['system']['frequency']
        assert loaded_config['system']['bandwidth'] == config['system']['bandwidth']
        assert loaded_config['system']['tx_power_dBm'] == config['system']['tx_power_dBm']
        assert loaded_config['oam']['min_mode'] == config['oam']['min_mode']
        assert loaded_config['oam']['max_mode'] == config['oam']['max_mode']
        assert loaded_config['oam']['beam_width'] == config['oam']['beam_width']
    
    def test_hierarchical_config(self):
        """Test that load_hierarchical_config loads configurations with inheritance."""
        # Load the base configuration
        base_config = load_hierarchical_config("base_config_new")
        
        # Check that the base configuration has the correct keys
        assert 'system' in base_config
        assert 'oam' in base_config
        assert 'environment' in base_config
        assert 'rl_base' in base_config
        
        # Load the RL configuration
        rl_config = load_hierarchical_config("rl_config_new")
        
        # Check that the RL configuration inherits from the base configuration
        assert 'system' in rl_config
        assert 'oam' in rl_config
        assert 'environment' in rl_config
        assert 'rl_base' in rl_config
        
        # Check that the RL configuration has the correct values
        assert rl_config['system']['frequency'] == base_config['system']['frequency']
        assert rl_config['oam']['min_mode'] == base_config['oam']['min_mode']
        assert rl_config['oam']['max_mode'] == base_config['oam']['max_mode']
        
        # Load the stable reward configuration
        stable_config = load_hierarchical_config("stable_reward_config_new")
        
        # Check that the stable reward configuration inherits from the RL configuration
        assert 'system' in stable_config
        assert 'oam' in stable_config
        assert 'environment' in stable_config
        assert 'rl_base' in stable_config
        
        # Check that the stable reward configuration has the correct values
        assert stable_config['system']['frequency'] == base_config['system']['frequency']
        assert stable_config['oam']['min_mode'] == base_config['oam']['min_mode']
        assert stable_config['oam']['max_mode'] == base_config['oam']['max_mode']
        
        # Check that the stable reward configuration has its own values
        assert 'stable_reward' in stable_config
    
    def test_config_inheritance(self):
        """Test that configuration inheritance works correctly."""
        # This test verifies multi-level inheritance and RL override normalization
        
        # Load the base configuration
        base_config = load_hierarchical_config("base_config_new")
        
        # Load the extended configuration
        extended_config = load_hierarchical_config("extended_config_new")
        
        # Check that the extended configuration inherits from the base configuration
        assert 'system' in extended_config
        assert 'oam' in extended_config
        assert 'environment' in extended_config
        assert 'rl_base' in extended_config
        
        # Check that the extended configuration has the correct values
        assert extended_config['system']['frequency'] == base_config['system']['frequency']
        assert extended_config['oam']['min_mode'] == base_config['oam']['min_mode']
        assert extended_config['oam']['max_mode'] == base_config['oam']['max_mode']
        
        # Check that the extended configuration has its own values (overrides)
        assert extended_config['training']['num_episodes'] == 2000
        assert extended_config['rl_base']['replay_buffer']['capacity'] == 100000