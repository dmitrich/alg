"""
Tests for TensorBoard configuration loader.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch
from utils_tensorboard.config import (
    load_tensorboard_config,
    get_runs_directory,
    get_default_port,
    get_max_port_attempts
)


class TestTensorBoardConfig:
    """Unit tests for TensorBoard configuration loader."""
    
    def test_load_default_config(self):
        """Test loading default configuration when file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            config = load_tensorboard_config("nonexistent.json")
            
            assert config["runs_directory"] == "runs"
            assert config["default_port"] == 6006
            assert config["max_port_attempts"] == 10
    
    def test_load_custom_config(self):
        """Test loading custom configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            custom_config = {
                "runs_directory": "custom_runs",
                "default_port": 7007,
                "max_port_attempts": 5
            }
            json.dump(custom_config, f)
            config_path = f.name
        
        try:
            config = load_tensorboard_config(config_path)
            
            assert config["runs_directory"] == "custom_runs"
            assert config["default_port"] == 7007
            assert config["max_port_attempts"] == 5
        finally:
            os.unlink(config_path)
    
    def test_load_partial_config(self):
        """Test loading partial configuration merges with defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            partial_config = {
                "runs_directory": "my_runs"
            }
            json.dump(partial_config, f)
            config_path = f.name
        
        try:
            config = load_tensorboard_config(config_path)
            
            # Custom value
            assert config["runs_directory"] == "my_runs"
            # Default values
            assert config["default_port"] == 6006
            assert config["max_port_attempts"] == 10
        finally:
            os.unlink(config_path)
    
    def test_load_config_handles_invalid_json(self):
        """Test graceful handling of invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json {")
            config_path = f.name
        
        try:
            with patch('builtins.print'):
                config = load_tensorboard_config(config_path)
                
                # Should return defaults
                assert config["runs_directory"] == "runs"
                assert config["default_port"] == 6006
                assert config["max_port_attempts"] == 10
        finally:
            os.unlink(config_path)
    
    def test_get_runs_directory(self):
        """Test get_runs_directory helper function."""
        runs_dir = get_runs_directory()
        assert isinstance(runs_dir, str)
        assert len(runs_dir) > 0
    
    def test_get_default_port(self):
        """Test get_default_port helper function."""
        port = get_default_port()
        assert isinstance(port, int)
        assert port > 0
    
    def test_get_max_port_attempts(self):
        """Test get_max_port_attempts helper function."""
        attempts = get_max_port_attempts()
        assert isinstance(attempts, int)
        assert attempts > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
