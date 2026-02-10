"""
Tests for TensorBoard instructions printer.
"""

import pytest
from unittest.mock import patch
from utils_tensorboard.instructions import print_tensorboard_instructions


class TestTensorBoardInstructions:
    """Unit tests for TensorBoard instructions printer."""
    
    def test_print_instructions_includes_log_dir(self):
        """Test that instructions include the log directory path."""
        log_dir = "/path/to/runs/2026-02-09_001/logs/tensorboard"
        
        with patch('builtins.print') as mock_print:
            print_tensorboard_instructions(log_dir)
            
            # Collect all print calls
            print_calls = [str(call) for call in mock_print.call_args_list]
            all_output = ' '.join(print_calls)
            
            # Verify log directory is mentioned
            assert log_dir in all_output
    
    def test_print_instructions_includes_tensorboard_command(self):
        """Test that instructions include the tensorboard command."""
        log_dir = "/path/to/runs/2026-02-09_001/logs/tensorboard"
        
        with patch('builtins.print') as mock_print:
            print_tensorboard_instructions(log_dir)
            
            print_calls = [str(call) for call in mock_print.call_args_list]
            all_output = ' '.join(print_calls)
            
            # Verify tensorboard command is present
            assert 'tensorboard --logdir=' in all_output
            assert '--port=' in all_output
    
    def test_print_instructions_includes_runs_directory(self):
        """Test that instructions include the runs directory name."""
        log_dir = "/path/to/runs/2026-02-09_001/logs/tensorboard"
        
        with patch('builtins.print') as mock_print:
            print_tensorboard_instructions(log_dir)
            
            print_calls = [str(call) for call in mock_print.call_args_list]
            all_output = ' '.join(print_calls)
            
            # Verify runs directory is mentioned
            assert 'runs' in all_output or 'Default runs directory' in all_output
    
    def test_print_instructions_includes_browser_url(self):
        """Test that instructions include the browser URL."""
        log_dir = "/path/to/runs/2026-02-09_001/logs/tensorboard"
        
        with patch('builtins.print') as mock_print:
            print_tensorboard_instructions(log_dir)
            
            print_calls = [str(call) for call in mock_print.call_args_list]
            all_output = ' '.join(print_calls)
            
            # Verify browser URL is present
            assert 'http://localhost:' in all_output
    
    def test_print_instructions_includes_config_file(self):
        """Test that instructions mention the config file location."""
        log_dir = "/path/to/runs/2026-02-09_001/logs/tensorboard"
        
        with patch('builtins.print') as mock_print:
            print_tensorboard_instructions(log_dir)
            
            print_calls = [str(call) for call in mock_print.call_args_list]
            all_output = ' '.join(print_calls)
            
            # Verify config file is mentioned
            assert 'tensorboard.json' in all_output
    
    def test_print_instructions_formatted_nicely(self):
        """Test that instructions are formatted with separators."""
        log_dir = "/path/to/runs/2026-02-09_001/logs/tensorboard"
        
        with patch('builtins.print') as mock_print:
            print_tensorboard_instructions(log_dir)
            
            print_calls = [str(call) for call in mock_print.call_args_list]
            all_output = ' '.join(print_calls)
            
            # Verify formatting elements are present
            assert '=' in all_output  # Separator lines
            assert 'TensorBoard' in all_output  # Title


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
