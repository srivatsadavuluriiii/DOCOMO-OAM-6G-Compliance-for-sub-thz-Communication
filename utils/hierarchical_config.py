#!/usr/bin/env python3
"""
Hierarchical configuration utility for OAM 6G.

This module handles hierarchical configuration inheritance to eliminate
redundancy and ensure parameter consistency across all configurations.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from .input_sanitizer import sanitized_config_loader


class HierarchicalConfig:
    """
    Hierarchical configuration manager for OAM 6G.
    
    This class handles configuration inheritance to eliminate redundancy
    and ensure parameter consistency across all configurations.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the hierarchical configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.base_config = None
        self.config_cache = {}
        
    def load_base_config(self) -> Dict[str, Any]:
        """
        Load the base configuration with sanitization.
        
        Returns:
            Base configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid or malicious
        """
        base_path = self.config_dir / "base_config_new.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"Base configuration not found: {base_path}")
        
                                     
        self.base_config = sanitized_config_loader(str(base_path))
        self.base_config = self._ensure_legacy_compatibility(self.base_config)
        
        return self.base_config
    
    def load_config_with_inheritance(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration with inheritance from base and sanitization.
        
        Args:
            config_name: Name of the configuration file (without .yaml)
            
        Returns:
            Merged configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid or malicious
        """
                           
        if config_name in self.config_cache:
            return self.config_cache[config_name]

                                                
        if self.base_config is None:
            self.load_base_config()

                                                               
        chain = self._get_inheritance_chain(config_name)
                                                       
        if chain[-1] != config_name:
            chain.append(config_name)

                                   
        merged_config: Dict[str, Any] = self.base_config.copy()

                                              
        for name in chain[1:]:
            config_path = self.config_dir / f"{name}.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration not found: {config_path}")
            layer_cfg = sanitized_config_loader(str(config_path))
            merged_config = self._deep_merge(merged_config, layer_cfg)

                                                                          
        merged_config = self._normalize_rl_sections(merged_config)

                                                                        
        merged_config = self._ensure_legacy_compatibility(merged_config)

                          
        self.config_cache[config_name] = merged_config

        return merged_config

    def _normalize_rl_sections(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize RL-related sections provided at the top level in child configs
        so that they override values under rl_base.*.

        We do not remove the top-level keys to preserve backward compatibility
        for code that might read them directly. This function only ensures that
        rl_base.* reflects the intended overrides.
        """
        rl_sections = [
            'network', 'replay_buffer', 'exploration', 'evaluation', 'reward'
        ]

        if 'rl_base' not in config or not isinstance(config['rl_base'], dict):
                                                       
            return config

        for section in rl_sections:
            top_value = config.get(section)
            if isinstance(top_value, dict):
                base_section = config['rl_base'].get(section, {})
                if not isinstance(base_section, dict):
                    base_section = {}
                                                        
                merged_section = self._deep_merge(base_section, top_value)
                config['rl_base'][section] = merged_section

        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration for consistency.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
                                 
        required_sections = ['system', 'oam', 'environment', 'mobility']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
                                    
        if 'oam' in config:
            oam = config['oam']
            if 'min_mode' in oam and 'max_mode' in oam:
                try:
                    min_mode = int(oam['min_mode'])
                    max_mode = int(oam['max_mode'])
                    if min_mode >= max_mode:
                        errors.append("min_mode must be less than max_mode")
                    if min_mode < 1:
                        errors.append("min_mode must be at least 1")
                except (ValueError, TypeError):
                    errors.append("min_mode and max_mode must be integers")
        
                                                                        
        if 'system' in config:
            system = config['system']
                                                                     
            if 'frequency' in system:
                try:
                    freq = float(system['frequency'])
                    if freq <= 0:
                        errors.append("frequency must be positive")
                except (ValueError, TypeError):
                    errors.append("frequency must be a number")
            if 'bandwidth' in system:
                try:
                    bw = float(system['bandwidth'])
                    if bw <= 0:
                        errors.append("bandwidth must be positive")
                except (ValueError, TypeError):
                    errors.append("bandwidth must be a number")
            if 'tx_power_dBm' in system:
                try:
                    if isinstance(system['tx_power_dBm'], dict):
                        for _, p in system['tx_power_dBm'].items():
                            pval = float(p)
                            if pval < -20 or pval > 50:
                                errors.append("tx_power_dBm band value must be between -20 and 50 dBm")
                    else:
                        power = float(system['tx_power_dBm'])
                        if power < -20 or power > 50:
                            errors.append("tx_power_dBm must be between -20 and 50 dBm")
                except (ValueError, TypeError):
                    errors.append("tx_power_dBm must be a number or dict of numbers")
        
                                              
        if 'training' in config:
            training = config['training']
            if 'num_episodes' in training:
                try:
                    episodes = float(training['num_episodes'])
                    if episodes <= 0:
                        errors.append("num_episodes must be positive")
                except (ValueError, TypeError):
                    errors.append("num_episodes must be a number")
            if 'batch_size' in training:
                try:
                    batch_size = float(training['batch_size'])
                    if batch_size <= 0:
                        errors.append("batch_size must be positive")
                except (ValueError, TypeError):
                    errors.append("batch_size must be a number")
            if 'learning_rate' in training:
                try:
                    lr = float(training['learning_rate'])
                    if lr <= 0:
                        errors.append("learning_rate must be positive")
                except (ValueError, TypeError):
                    errors.append("learning_rate must be a number")
        
        return errors
    
    def get_config_info(self, config_name: str) -> Dict[str, Any]:
        """
        Get information about a configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration information
        """
        config = self.load_config_with_inheritance(config_name)
        
                                     
        section_counts = {}
        for section, params in config.items():
            if isinstance(params, dict):
                section_counts[section] = len(params)
        
                               
        inheritance_chain = self._get_inheritance_chain(config_name)
        
        return {
            'name': config_name,
            'inheritance_chain': inheritance_chain,
            'section_counts': section_counts,
            'total_sections': len(config),
            'has_training': 'training' in config,
            'has_reward': 'reward' in config,
            'has_stable_reward': 'stable_reward' in config
        }

    def _ensure_legacy_compatibility(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add legacy scalar keys (frequency, bandwidth, tx_power_dBm) if only banded
        forms are present, defaulting to the mmwave band for compatibility.
        """
        cfg = config.copy()
        system = cfg.get('system', {})
        changed = False
                   
        if 'frequency' not in system and isinstance(system.get('frequency_bands'), dict):
            mmwave = system['frequency_bands'].get('mmwave')
            if mmwave is not None:
                system['frequency'] = mmwave
                changed = True
                   
        if 'bandwidth' not in system and isinstance(system.get('bandwidth_bands'), dict):
            mmwave_bw = system['bandwidth_bands'].get('mmwave')
            if mmwave_bw is not None:
                system['bandwidth'] = mmwave_bw
                changed = True
                                
        if isinstance(system.get('tx_power_dBm'), dict):
            mmwave_tx = system['tx_power_dBm'].get('mmwave')
            if mmwave_tx is not None:
                system['tx_power_dBm_scalar'] = mmwave_tx                         
                changed = True
        if changed:
            cfg['system'] = system
        return cfg
    
    def _get_inheritance_chain(self, config_name: str) -> List[str]:
        """
        Get the inheritance chain for a configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            List of configuration names in inheritance order
        """
        chain = ['base_config_new']
        
        if config_name == 'rl_config_new':
            chain.append('rl_config_new')
        elif config_name == 'stable_reward_config_new':
            chain.extend(['rl_config_new', 'stable_reward_config_new'])
        elif config_name == 'extended_config_new':
            chain.extend(['rl_config_new', 'extended_config_new'])
        
        return chain
    
    def list_configurations(self) -> List[Dict[str, Any]]:
        """
        List all available configurations.
        
        Returns:
            List of configuration information
        """
        configs = []
        
                                          
        for config_file in self.config_dir.glob("*_new.yaml"):
            config_name = config_file.stem
            if config_name != "base_config_new":
                try:
                    info = self.get_config_info(config_name)
                    configs.append(info)
                except Exception as e:
                    configs.append({
                        'name': config_name,
                        'error': str(e)
                    })
        
        return configs
    
    def export_flat_config(self, config_name: str, output_path: str) -> None:
        """
        Export a configuration as a flat file (no inheritance).
        
        Args:
            config_name: Name of the configuration
            output_path: Output file path
        """
        config = self.load_config_with_inheritance(config_name)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def compare_configs(self, config1: str, config2: str) -> Dict[str, Any]:
        """
        Compare two configurations.
        
        Args:
            config1: First configuration name
            config2: Second configuration name
            
        Returns:
            Comparison results
        """
        cfg1 = self.load_config_with_inheritance(config1)
        cfg2 = self.load_config_with_inheritance(config2)
        
                          
        differences = {}
        all_keys = set(cfg1.keys()) | set(cfg2.keys())
        
        for key in all_keys:
            if key not in cfg1:
                differences[key] = {'type': 'only_in_config2', 'value': cfg2[key]}
            elif key not in cfg2:
                differences[key] = {'type': 'only_in_config1', 'value': cfg1[key]}
            elif cfg1[key] != cfg2[key]:
                differences[key] = {
                    'type': 'different_values',
                    'config1_value': cfg1[key],
                    'config2_value': cfg2[key]
                }
        
        return {
            'config1': config1,
            'config2': config2,
            'differences': differences,
            'total_differences': len(differences)
        }


def load_hierarchical_config(config_name: str, config_dir: str = "config") -> Dict[str, Any]:
    """
    Convenience function to load a hierarchical configuration.
    
    Args:
        config_name: Name of the configuration (without .yaml)
        config_dir: Directory containing configuration files
        
    Returns:
        Merged configuration dictionary
    """
    config_manager = HierarchicalConfig(config_dir)
    return config_manager.load_config_with_inheritance(config_name)


def validate_hierarchical_config(config_name: str, config_dir: str = "config") -> List[str]:
    """
    Convenience function to validate a hierarchical configuration.
    
    Args:
        config_name: Name of the configuration (without .yaml)
        config_dir: Directory containing configuration files
        
    Returns:
        List of validation errors
    """
    config_manager = HierarchicalConfig(config_dir)
    config = config_manager.load_config_with_inheritance(config_name)
    return config_manager.validate_config(config) 
