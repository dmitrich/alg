"""
Tests for TensorBoard server launcher functionality.
Includes unit tests for port detection and server launching.
"""

import pytest
import socket
import tempfile
import os
from unittest.mock import patch, MagicMock
from utils_tensorboard.server import is_port_in_use, is_tensorboard_running, launch_tensorboard_server


class TestPortDetection:
    """Unit tests for port detection functionality."""
    
    def test_is_port_in_use_with_free_port(self):
        """Test that is_port_in_use returns False for a free port."""
        # Use a high port number that's unlikely to be in use
        port = 54321
        assert not is_port_in_use(port)
    
    def test_is_port_in_use_with_occupied_port(self):
        """Test that is_port_in_use returns True for an occupied port."""
        # Create a socket and bind to a port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))  # Bind to any available port
            port = s.getsockname()[1]  # Get the port number
            
            # Port should be in use while socket is open
            assert is_port_in_use(port)
        
        # Port should be free after socket is closed
        assert not is_port_in_use(port)
    
    def test_is_tensorboard_running_when_not_running(self):
        """Test that is_tensorboard_running returns False when TensorBoard is not running."""
        # Use a port that's unlikely to have TensorBoard running
        port = 54322
        assert not is_tensorboard_running(port)
    
    def test_is_tensorboard_running_when_running(self):
        """Test that is_tensorboard_running returns True when TensorBoard is running."""
        # Mock urllib.request.urlopen to simulate TensorBoard responding
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        
        with patch('urllib.request.urlopen', return_value=mock_response):
            assert is_tensorboard_running(6006)


class TestServerLauncher:
    """Unit tests for TensorBoard server launcher."""
    
    def test_launch_when_already_running(self):
        """Test that launcher detects when TensorBoard is already running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock is_tensorboard_running to return True
            with patch('utils_tensorboard.server.is_tensorboard_running', return_value=True):
                with patch('builtins.print') as mock_print:
                    launch_tensorboard_server(log_dir, start_port=6006)
                    
                    # Verify message about already running
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    all_output = ' '.join(print_calls)
                    assert 'already running' in all_output
                    assert 'http://localhost:6006/' in all_output
    
    def test_launch_with_tensorboard_not_installed(self):
        """Test graceful handling when TensorBoard is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock tensorboard import to fail
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if name == 'tensorboard':
                    raise ImportError("Mocked import failure")
                return real_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                with patch('builtins.print') as mock_print:
                    # Should not raise an exception
                    launch_tensorboard_server(log_dir)
                    
                    # Should print helpful message
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    all_output = ' '.join(print_calls)
                    assert 'TensorBoard not installed' in all_output
                    assert 'uv pip install tensorboard' in all_output
    
    def test_launch_on_available_port(self):
        """Test successful launch on an available port."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock is_tensorboard_running to return False (not already running)
            with patch('utils_tensorboard.server.is_tensorboard_running', return_value=False):
                # Mock is_port_in_use to return False (port available)
                with patch('utils_tensorboard.server.is_port_in_use', return_value=False):
                    # Mock subprocess.Popen to avoid actually launching TensorBoard
                    with patch('subprocess.Popen') as mock_popen:
                        with patch('builtins.print') as mock_print:
                            launch_tensorboard_server(log_dir, start_port=6006, max_attempts=1)
                            
                            # Verify Popen was called
                            assert mock_popen.call_count == 1
                            
                            # Verify success message was printed
                            print_calls = [str(call) for call in mock_print.call_args_list]
                            all_output = ' '.join(print_calls)
                            assert 'launched successfully' in all_output
                            assert 'http://localhost:' in all_output
    
    def test_launch_with_port_retry(self):
        """Test that launcher retries on next port when current port is in use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock is_tensorboard_running to return False (not already running)
            with patch('utils_tensorboard.server.is_tensorboard_running', return_value=False):
                # Mock is_port_in_use to simulate first port being in use
                call_count = [0]
                
                def mock_is_port_in_use(port):
                    call_count[0] += 1
                    # First call (port 6006) returns True (in use)
                    # Second call (port 6007) returns False (available)
                    return call_count[0] == 1
                
                with patch('utils_tensorboard.server.is_port_in_use', side_effect=mock_is_port_in_use):
                    with patch('subprocess.Popen') as mock_popen:
                        with patch('builtins.print') as mock_print:
                            launch_tensorboard_server(log_dir, start_port=6006, max_attempts=2)
                            
                            # Verify Popen was called (server launched on second port)
                            assert mock_popen.call_count == 1
                            
                            # Verify retry message was printed
                            print_calls = [str(call) for call in mock_print.call_args_list]
                            all_output = ' '.join(print_calls)
                            assert 'Port 6006 in use' in all_output
                            assert 'trying 6007' in all_output
    
    def test_launch_failure_after_max_attempts(self):
        """Test fallback to manual command after all ports are in use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock is_tensorboard_running to return False (not already running)
            with patch('utils_tensorboard.server.is_tensorboard_running', return_value=False):
                # Mock is_port_in_use to always return True (all ports in use)
                with patch('utils_tensorboard.server.is_port_in_use', return_value=True):
                    with patch('builtins.print') as mock_print:
                        launch_tensorboard_server(log_dir, start_port=6006, max_attempts=3)
                        
                        # Verify fallback message was printed
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        all_output = ' '.join(print_calls)
                        assert 'Could not launch TensorBoard automatically' in all_output
                        assert 'To view logs manually' in all_output
                        assert f'tensorboard --logdir={log_dir}' in all_output
    
    def test_launch_handles_subprocess_exception(self):
        """Test graceful handling of subprocess launch exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'logs', 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            
            # Mock is_tensorboard_running to return False (not already running)
            with patch('utils_tensorboard.server.is_tensorboard_running', return_value=False):
                # Mock subprocess.Popen to raise an exception
                with patch('subprocess.Popen', side_effect=RuntimeError("Test error")):
                    with patch('builtins.print') as mock_print:
                        # Should not raise an exception
                        launch_tensorboard_server(log_dir, start_port=6006, max_attempts=2)
                        
                        # Should print error and fallback message
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        all_output = ' '.join(print_calls)
                        assert 'Failed to launch TensorBoard' in all_output or 'Could not launch TensorBoard' in all_output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
