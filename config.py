"""
Configuration module for GPT refactoring project.
Handles loading and validation of hyperparameters from config.json.
"""

import json  # Added for configuration file I/O


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
    # Added for file I/O: Open and read configuration file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # Added for error handling: Provide clear message for missing file
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Please create config.json with required hyperparameters."
        )
    except json.JSONDecodeError as e:
        # Added for error handling: Provide clear message for invalid JSON
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
    # Added for validation: Define required configuration keys
    required_keys = [
        'batch_size', 'block_size', 'max_iters', 'eval_interval',
        'learning_rate', 'eval_iters', 'n_embd', 'n_head', 'n_layer', 'dropout'
    ]
    
    # Added for validation: Check all required keys are present
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required configuration parameters: {', '.join(missing_keys)}"
        )
    
    # Added for validation: Validate batch_size > 0
    if config['batch_size'] <= 0:
        raise ValueError(
            f"batch_size must be greater than 0, got {config['batch_size']}"
        )
    
    # Added for validation: Validate block_size > 0
    if config['block_size'] <= 0:
        raise ValueError(
            f"block_size must be greater than 0, got {config['block_size']}"
        )
    
    # Added for validation: Validate max_iters > 0
    if config['max_iters'] <= 0:
        raise ValueError(
            f"max_iters must be greater than 0, got {config['max_iters']}"
        )
    
    # Added for validation: Validate eval_interval > 0
    if config['eval_interval'] <= 0:
        raise ValueError(
            f"eval_interval must be greater than 0, got {config['eval_interval']}"
        )
    
    # Added for validation: Validate learning_rate > 0
    if config['learning_rate'] <= 0:
        raise ValueError(
            f"learning_rate must be greater than 0, got {config['learning_rate']}"
        )
    
    # Added for validation: Validate eval_iters > 0
    if config['eval_iters'] <= 0:
        raise ValueError(
            f"eval_iters must be greater than 0, got {config['eval_iters']}"
        )
    
    # Added for validation: Validate n_embd > 0
    if config['n_embd'] <= 0:
        raise ValueError(
            f"n_embd must be greater than 0, got {config['n_embd']}"
        )
    
    # Added for validation: Validate n_head > 0
    if config['n_head'] <= 0:
        raise ValueError(
            f"n_head must be greater than 0, got {config['n_head']}"
        )
    
    # Added for validation: Validate n_embd is divisible by n_head
    if config['n_embd'] % config['n_head'] != 0:
        raise ValueError(
            f"n_embd ({config['n_embd']}) must be divisible by n_head ({config['n_head']})"
        )
    
    # Added for validation: Validate n_layer > 0
    if config['n_layer'] <= 0:
        raise ValueError(
            f"n_layer must be greater than 0, got {config['n_layer']}"
        )
    
    # Added for validation: Validate 0 <= dropout < 1
    if not (0 <= config['dropout'] < 1):
        raise ValueError(
            f"dropout must be in range [0, 1), got {config['dropout']}"
        )
    
    # Added for validation: Return True if all validations pass
    return True
