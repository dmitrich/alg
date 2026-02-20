"""
Configuration module for GPT refactoring project.
Handles loading and validation of hyperparameters from config.json.
"""

import json

def load_config(config_path: str = 'config.json') -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Dictionary containing hyperparameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config contains invalid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Please create config.json with required hyperparameters."
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in configuration file {config_path}: {e}"
        )

def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If any parameter is invalid
    """
    required_keys = [
        'batch_size', 'block_size', 'max_iters', 'eval_interval',
        'learning_rate', 'eval_iters', 'n_embd', 'n_head', 'n_layer', 'dropout'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required configuration parameters: {', '.join(missing_keys)}"
        )
    
    if config['batch_size'] <= 0:
        raise ValueError(
            f"batch_size must be greater than 0, got {config['batch_size']}"
        )
    
    if config['block_size'] <= 0:
        raise ValueError(
            f"block_size must be greater than 0, got {config['block_size']}"
        )
    
    if config['max_iters'] <= 0:
        raise ValueError(
            f"max_iters must be greater than 0, got {config['max_iters']}"
        )
    
    if config['eval_interval'] <= 0:
        raise ValueError(
            f"eval_interval must be greater than 0, got {config['eval_interval']}"
        )
    
    if config['learning_rate'] <= 0:
        raise ValueError(
            f"learning_rate must be greater than 0, got {config['learning_rate']}"
        )
    
    if config['eval_iters'] <= 0:
        raise ValueError(
            f"eval_iters must be greater than 0, got {config['eval_iters']}"
        )
    
    if config['n_embd'] <= 0:
        raise ValueError(
            f"n_embd must be greater than 0, got {config['n_embd']}"
        )
    
    if config['n_head'] <= 0:
        raise ValueError(
            f"n_head must be greater than 0, got {config['n_head']}"
        )
    
    if config['n_embd'] % config['n_head'] != 0:
        raise ValueError(
            f"n_embd ({config['n_embd']}) must be divisible by n_head ({config['n_head']})"
        )
    
    if config['n_layer'] <= 0:
        raise ValueError(
            f"n_layer must be greater than 0, got {config['n_layer']}"
        )
    
    if not (0 <= config['dropout'] < 1):
        raise ValueError(
            f"dropout must be in range [0, 1), got {config['dropout']}"
        )
    
    return True
