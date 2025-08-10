import os
import yaml
from typing import Dict, Any
from .input_sanitizer import sanitized_config_loader


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with sanitization.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration parameters
        
    Raises:
        ValueError: If configuration is invalid or malicious
    """
    return sanitized_config_loader(config_path)


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Dictionary containing the configuration parameters
        save_path: Path to save the YAML configuration file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary that overrides base values
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in merged_config and 
            isinstance(merged_config[key], dict) and 
            isinstance(value, dict)
        ):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config 