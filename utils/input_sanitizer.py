#!/usr/bin/env python3
"""
Input sanitization system for OAM 6G configuration files.

This module provides comprehensive input sanitization to prevent
malicious configurations, validate YAML structure, check parameter types,
and ensure values are within reasonable bounds.
"""

import yaml
import re
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging

                   
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputSanitizer:
    """
    Comprehensive input sanitization for configuration files.
    
    Provides protection against:
    - Malicious YAML content
    - Invalid parameter types
    - Out-of-range values
    - Malformed configuration structure
    """
    
    def __init__(self):
        """Initialize the input sanitizer."""
        self.allowed_sections = {
            'system', 'oam', 'environment', 'mobility', 'training', 
            'rl_base', 'exploration', 'evaluation', 'network', 'replay_buffer',
            'enhanced_params', 'reward', 'stable_reward',                           
            'distance_optimization', 'distance_thresholds', 'mode_preferences', 
            'optimization_weights', 'adaptive_parameters', 'distance_reward',
            'performance_tracking', 'distance_categories', 'optimization_strategies',
            'distance_sinr_thresholds', 'handover_optimization',                                  
                                
            'hybrid_system', 'distance_optimization_bands', 'physics'
        }
        
        self.parameter_specs = {
            'system': {
                'frequency': {'type': (int, float), 'range': (1e9, 1e12), 'required': False},                     
                'bandwidth': {'type': (int, float), 'range': (1e6, 10e9), 'required': False},                       
                                                                        
                'tx_power_dBm': {'type': (int, float, dict), 'range': (-20, 50), 'required': False},
                'noise_figure_dB': {'type': (int, float), 'range': (1, 20), 'required': True},
                'noise_temp': {'type': (int, float), 'range': (100, 500), 'required': True},
                                      
                'frequency_bands': {'type': dict, 'range': None, 'required': False},
                'bandwidth_bands': {'type': dict, 'range': None, 'required': False},
                'max_throughput_gbps': {'type': dict, 'range': None, 'required': False},
                'tx_antenna_gain_dBi': {'type': (int, float, dict), 'range': None, 'required': False},
                'rx_antenna_gain_dBi': {'type': (int, float, dict), 'range': None, 'required': False}
            },
            'oam': {
                'min_mode': {'type': int, 'range': (1, 10), 'required': True},
                'max_mode': {'type': int, 'range': (2, 12), 'required': True},
                'beam_width': {'type': (int, float), 'range': (0.001, 0.1), 'required': True},
                'mode_spacing': {'type': int, 'range': (1, 5), 'required': False},
                                      
                'beam_width_bands': {'type': dict, 'range': None, 'required': False}
            },
            'environment': {
                'humidity': {'type': (int, float), 'range': (0, 100), 'required': True},
                'temperature': {'type': (int, float), 'range': (-50, 100), 'required': True},
                'pressure': {'type': (int, float), 'range': (50, 120), 'required': True},
                'turbulence_strength': {'type': (int, float), 'range': (1e-20, 1e-10), 'required': True},
                'pointing_error_std': {'type': (int, float), 'range': (0.001, 0.1), 'required': True},
                'rician_k_factor': {'type': (int, float), 'range': (0.1, 20), 'required': True}
            },
            'mobility': {
                'max_speed': {'type': (int, float), 'range': (0, 100), 'required': False},
                'update_interval': {'type': (int, float), 'range': (0.1, 10), 'required': False},
                'min_speed': {'type': (int, float), 'range': (0, 100), 'required': False},
                'direction_change_prob': {'type': (int, float), 'range': (0, 1), 'required': False},
                                      
                'area_size_bands': {'type': dict, 'range': None, 'required': False}
            },
            'training': {
                'num_episodes': {'type': int, 'range': (1, 10000), 'required': False},
                'episodes': {'type': int, 'range': (1, 10000), 'required': False},                    
                'batch_size': {'type': int, 'range': (1, 1000), 'required': False},
                'learning_rate': {'type': (int, float), 'range': (1e-6, 1), 'required': False},
                'max_steps_per_episode': {'type': int, 'range': (1, 10000), 'required': False},
                'memory_size': {'type': int, 'range': (1000, 1000000), 'required': False},
                'target_update_interval': {'type': int, 'range': (1, 1000), 'required': False},
                'target_update_freq': {'type': int, 'range': (1, 1000), 'required': False},
                'gamma': {'type': (int, float), 'range': (0, 1), 'required': False},
                                               
                'exploration': {'type': dict, 'range': None, 'required': False},
                'band_exploration': {'type': dict, 'range': None, 'required': False}
            },
            'rl_base': {
                'state_dim': {'type': int, 'range': (1, 1000), 'required': False},
                'action_dim': {'type': int, 'range': (1, 100), 'required': False},
                'hidden_dim': {'type': int, 'range': (10, 1000), 'required': False},
                'network': {'type': dict, 'range': None, 'required': False},
                'replay_buffer': {'type': dict, 'range': None, 'required': False},
                'exploration': {'type': dict, 'range': None, 'required': False},
                'evaluation': {'type': dict, 'range': None, 'required': False}
            },
            'exploration': {
                'epsilon_start': {'type': (int, float), 'range': (0, 1), 'required': False},
                'epsilon_end': {'type': (int, float), 'range': (0, 1), 'required': False},
                'epsilon_decay': {'type': (int, float), 'range': (0.001, 1), 'required': False}
            },
            'network': {
                'hidden_layers': {'type': list, 'range': None, 'required': False},
                'activation': {'type': str, 'range': None, 'required': False}
            },
            'replay_buffer': {
                'buffer_capacity': {'type': int, 'range': (1000, 1000000), 'required': False},
                'capacity': {'type': int, 'range': (1000, 1000000), 'required': False},
                'min_samples_to_learn': {'type': int, 'range': (10, 100000), 'required': False}
            },
            'stable_reward': {
                'smoothing_factor': {'type': (int, float), 'range': (0, 1), 'required': False},
                'window_size': {'type': int, 'range': (1, 100), 'required': False},
                'reward_scale': {'type': (int, float), 'range': (0.1, 10), 'required': False},
                'reward_min': {'type': (int, float), 'range': (-100, 0), 'required': False},
                'reward_max': {'type': (int, float), 'range': (0, 100), 'required': False},
                'sinr_scaling_factor': {'type': (int, float), 'range': (0, 10), 'required': False}
            },
                                            
            'distance_optimization': {
                'distance_thresholds': {'type': dict, 'range': None, 'required': False},
                'mode_preferences': {'type': dict, 'range': None, 'required': False},
                'optimization_weights': {'type': dict, 'range': None, 'required': False},
                'adaptive_parameters': {'type': dict, 'range': None, 'required': False},
                'distance_reward': {'type': dict, 'range': None, 'required': False},
                'performance_tracking': {'type': dict, 'range': None, 'required': False},
                'distance_categories': {'type': dict, 'range': None, 'required': False},
                'optimization_strategies': {'type': dict, 'range': None, 'required': False},
                'distance_sinr_thresholds': {'type': dict, 'range': None, 'required': False},
                'handover_optimization': {'type': dict, 'range': None, 'required': False}
            },
            'distance_thresholds': {
                'near_threshold': {'type': (int, float), 'range': (10, 200), 'required': False},
                'medium_threshold': {'type': (int, float), 'range': (50, 500), 'required': False},
                'far_threshold': {'type': (int, float), 'range': (100, 1000), 'required': False}
            },
            'mode_preferences': {
                'near_modes': {'type': list, 'range': None, 'required': False},
                'medium_modes': {'type': list, 'range': None, 'required': False},
                'far_modes': {'type': list, 'range': None, 'required': False}
            },
            'optimization_weights': {
                'distance_weight': {'type': (int, float), 'range': (0, 1), 'required': False},
                'throughput_weight': {'type': (int, float), 'range': (0, 1), 'required': False},
                'stability_weight': {'type': (int, float), 'range': (0, 1), 'required': False}
            },
            'adaptive_parameters': {
                'enabled': {'type': bool, 'range': None, 'required': False},
                'learning_rate': {'type': (int, float), 'range': (0.001, 0.1), 'required': False},
                'min_threshold': {'type': (int, float), 'range': (10, 200), 'required': False},
                'max_threshold': {'type': (int, float), 'range': (100, 1000), 'required': False}
            },
            'distance_reward': {
                'distance_bonus_factor': {'type': (int, float), 'range': (0, 1), 'required': False},
                'distance_penalty_factor': {'type': (int, float), 'range': (0, 1), 'required': False},
                'optimization_threshold': {'type': (int, float), 'range': (0, 1), 'required': False}
            },
            'performance_tracking': {
                'enable_history': {'type': bool, 'range': None, 'required': False},
                'history_length': {'type': int, 'range': (100, 10000), 'required': False},
                'enable_adaptive_learning': {'type': bool, 'range': None, 'required': False}
            },
            'distance_categories': {
                'near': {'type': dict, 'range': None, 'required': False},
                'medium': {'type': dict, 'range': None, 'required': False},
                'far': {'type': dict, 'range': None, 'required': False}
            },
            'optimization_strategies': {
                'conservative': {'type': dict, 'range': None, 'required': False},
                'balanced': {'type': dict, 'range': None, 'required': False},
                'aggressive': {'type': dict, 'range': None, 'required': False}
            },
            'distance_sinr_thresholds': {
                'near': {'type': (int, float), 'range': (-20, 10), 'required': False},
                'medium': {'type': (int, float), 'range': (-20, 10), 'required': False},
                'far': {'type': (int, float), 'range': (-20, 10), 'required': False}
            },
            'handover_optimization': {
                'enable_distance_aware_handover': {'type': bool, 'range': None, 'required': False},
                'min_handover_interval': {'type': int, 'range': (1, 50), 'required': False},
                'handover_hysteresis': {'type': (int, float), 'range': (0, 10), 'required': False},
                'time_to_trigger_steps': {'type': int, 'range': (1, 100), 'required': False},
                'handover_hysteresis_db': {'type': (int, float), 'range': (0, 20), 'required': False},
                'distance_based_handover_threshold': {'type': bool, 'range': None, 'required': False}
            },
                                
            'hybrid_system': {
                'band_selection': {'type': dict, 'range': None, 'required': False},
                'distance_thresholds': {'type': dict, 'range': None, 'required': False},
                'performance_thresholds': {'type': dict, 'range': None, 'required': False},
                'environment_adaptation': {'type': dict, 'range': None, 'required': False}
            },
            'distance_optimization_bands': {
                'mmwave': {'type': dict, 'range': None, 'required': False},
                'sub_thz_low': {'type': dict, 'range': None, 'required': False},
                'sub_thz_high': {'type': dict, 'range': None, 'required': False}
            },
            'physics': {
                'atmospheric_absorption': {'type': dict, 'range': None, 'required': False},
                'atmospheric_absorption_dB_per_km': {'type': (int, float), 'range': (0, 1000), 'required': False},
                'path_loss_exponents': {'type': dict, 'range': None, 'required': False},
                'beamforming': {'type': dict, 'range': None, 'required': False}
            }
        }
        
                                      
        self.malicious_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'os\.system\s*\(',
            r'subprocess\s*\(',
            r'import\s+',
            r'from\s+.*\s+import',
            r'class\s+',
            r'def\s+',
            r'lambda\s+',
            r'\\x[0-9a-fA-F]{2}',               
            r'\\u[0-9a-fA-F]{4}',                   
            r'\\U[0-9a-fA-F]{8}',                            
        ]
    
    def sanitize_yaml_content(self, yaml_content: str) -> Tuple[bool, List[str]]:
        """
        Sanitize YAML content for malicious patterns.
        
        Args:
            yaml_content: Raw YAML content as string
            
        Returns:
            Tuple of (is_safe, error_messages)
        """
        errors = []
        
                                      
        for pattern in self.malicious_patterns:
            if re.search(pattern, yaml_content, re.IGNORECASE):
                errors.append(f"Malicious pattern detected: {pattern}")
        
                                                     
        max_depth = 0
        current_depth = 0
        for char in yaml_content:
            if char == ' ':
                current_depth += 1
            elif char == '\n':
                max_depth = max(max_depth, current_depth)
                current_depth = 0
        
        if max_depth > 50:                                  
            errors.append("Excessive nesting detected (potential DoS)")
        
                                        
        lines = yaml_content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 1000:                                     
                errors.append(f"Line {i+1} is too long ({len(line)} characters)")
        
                                       
        if len(yaml_content) > 100000:               
            errors.append(f"File too large ({len(yaml_content)} bytes)")
        
        return len(errors) == 0, errors
    
    def validate_parameter_type(self, value: Any, expected_type: Union[type, Tuple[type, ...]], 
                              param_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate parameter type.
        
        Args:
            value: Parameter value
            expected_type: Expected type or tuple of types
            param_name: Parameter name for error reporting
            
        Returns:
            Tuple of (is_valid, error_message)
        """
                                                                
                                                                                  
        if isinstance(value, str) and isinstance(expected_type, tuple) and (int in expected_type or float in expected_type):
            try:
                                         
                float_value = float(value)
                                                                                
                if int in expected_type and float_value.is_integer():
                    return True, None
                                                  
                if float in expected_type:
                    return True, None
            except (ValueError, TypeError):
                pass                                         
        
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                type_names = [t.__name__ for t in expected_type]
                return False, f"{param_name} must be one of: {', '.join(type_names)}"
            else:
                return False, f"{param_name} must be {expected_type.__name__}"
        
        return True, None
    
    def validate_parameter_range(self, value: Union[int, float, str], min_val: Union[int, float], 
                               max_val: Union[int, float], param_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate parameter range.
        
        Args:
            value: Parameter value (int, float, or string that can be converted to float)
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            param_name: Parameter name for error reporting
            
        Returns:
            Tuple of (is_valid, error_message)
        """
                                           
        if isinstance(value, str):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False, f"{param_name} must be a valid number"
        
        if value < min_val or value > max_val:
            return False, f"{param_name} must be between {min_val} and {max_val}"
        
        return True, None
    
    def validate_section(self, section_name: str, section_data: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration section.
        
        Args:
            section_name: Name of the section
            section_data: Section data dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
                                     
        if section_name not in self.allowed_sections:
            errors.append(f"Unknown section: {section_name}")
            return errors
        
                                                       
        if section_name not in self.parameter_specs:
            return errors                                        
        
        specs = self.parameter_specs[section_name]
        
        for param_name, param_value in section_data.items():
                                         
            if param_name not in specs:
                errors.append(f"Unknown parameter in {section_name}: {param_name}")
                continue
            
            spec = specs[param_name]
            
                             
            is_valid_type, type_error = self.validate_parameter_type(
                param_value, spec['type'], f"{section_name}.{param_name}"
            )
            if not is_valid_type:
                errors.append(type_error)
                continue
            
                                             
            if spec['range'] is not None and isinstance(param_value, (int, float)):
                min_val, max_val = spec['range']
                is_valid_range, range_error = self.validate_parameter_range(
                    param_value, min_val, max_val, f"{section_name}.{param_name}"
                )
                if not is_valid_range:
                    errors.append(range_error)
            
                                          
            if spec['type'] == list and isinstance(param_value, list):
                if len(param_value) > 100:                   
                    errors.append(f"{section_name}.{param_name} list too large ({len(param_value)} items)")
        
                                       
        for param_name, spec in specs.items():
            if spec.get('required', False) and param_name not in section_data:
                errors.append(f"Required parameter missing in {section_name}: {param_name}")
        
        return errors
    
    def sanitize_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Sanitize a configuration dictionary.
        
        Args:
            config: Configuration dictionary to sanitize
            
        Returns:
            Tuple of (sanitized_config, error_messages)
        """
        errors = []
        sanitized_config = {}
        
                               
        for section_name, section_data in config.items():
            if not isinstance(section_data, dict):
                errors.append(f"Section {section_name} must be a dictionary")
                continue
            
            section_errors = self.validate_section(section_name, section_data)
            errors.extend(section_errors)
            
                                               
            if not section_errors:
                sanitized_config[section_name] = section_data.copy()
        
        return sanitized_config, errors
    
    def load_and_sanitize_yaml(self, file_path: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Load and sanitize a YAML configuration file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Tuple of (sanitized_config, error_messages)
        """
        errors = []
        
                              
        if not os.path.exists(file_path):
            return {}, [f"File not found: {file_path}"]
        
                         
        file_size = os.path.getsize(file_path)
        if file_size > 100000:               
            return {}, [f"File too large: {file_size} bytes"]
        
        try:
                               
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            
                                   
            is_safe, content_errors = self.sanitize_yaml_content(yaml_content)
            if not is_safe:
                return {}, content_errors
            
                        
            config = yaml.safe_load(yaml_content)
            if config is None:
                return {}, ["Empty or invalid YAML file"]
            
                                              
            if not isinstance(config, dict):
                return {}, ["Configuration must be a dictionary"]
            
                                    
            sanitized_config, config_errors = self.sanitize_config(config)
            errors.extend(config_errors)
            
            return sanitized_config, errors
            
        except yaml.YAMLError as e:
            return {}, [f"YAML parsing error: {str(e)}"]
        except Exception as e:
            return {}, [f"Unexpected error: {str(e)}"]

def create_sanitized_config_loader():
    """
    Create a sanitized configuration loader.
    
    Returns:
        Function that loads and sanitizes configurations
    """
    sanitizer = InputSanitizer()
    
    def load_sanitized_config(file_path: str) -> Dict[str, Any]:
        """
        Load and sanitize a configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Sanitized configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid or malicious
        """
        config, errors = sanitizer.load_and_sanitize_yaml(file_path)
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Configuration loaded and sanitized successfully: {file_path}")
        return config
    
    return load_sanitized_config

                                  
sanitized_config_loader = create_sanitized_config_loader() 
