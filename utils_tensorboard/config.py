"""
TensorBoard configuration loader.

This module provides functionality to load TensorBoard settings from
the configuration file.
"""

import json
import os


def load_tensorboard_config(config_path: str = "configs/tensorboard.json") -> dict:
    """
    Load TensorBoard configuration from JSON file.
    
    Args:
        config_path: Path to the TensorBoard configuration file
    
    Returns:
        Dictionary containing TensorBoard configuration settings
        
    Default configuration:
        {
            "runs_directory": "runs",
            "default_port": 6006,
            "max_port_attempts": 10,
            "histogram_interval": 100
        }
    """
    default_config = {
        "runs_directory": "runs",
        "default_port": 6006,
        "max_port_attempts": 10,
        "histogram_interval": 100
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**default_config, **config}
        else:
            print(f"TensorBoard config not found at {config_path}, using defaults")
            return default_config
    except Exception as e:
        print(f"Warning: Failed to load TensorBoard config: {e}")
        print("Using default configuration")
        return default_config


def get_runs_directory() -> str:
    """
    Get the configured runs directory name.
    
    Returns:
        The runs directory name from configuration (default: "runs")
    """
    config = load_tensorboard_config()
    return config.get("runs_directory", "runs")


def get_default_port() -> int:
    """
    Get the configured default TensorBoard port.
    
    Returns:
        The default port number from configuration (default: 6006)
    """
    config = load_tensorboard_config()
    return config.get("default_port", 6006)


def get_max_port_attempts() -> int:
    """
    Get the configured maximum port retry attempts.
    
    Returns:
        The maximum number of port attempts from configuration (default: 10)
    """
    config = load_tensorboard_config()
    return config.get("max_port_attempts", 10)


def get_histogram_interval() -> int:
    """
    Get the configured histogram logging interval.
    
    Returns:
        The number of steps between histogram logging (default: 100)
    """
    config = load_tensorboard_config()
    return config.get("histogram_interval", 100)
