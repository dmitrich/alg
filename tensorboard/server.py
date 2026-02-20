"""
TensorBoard server launcher with automatic port retry logic.

This module provides functionality to automatically launch TensorBoard as a
background process with intelligent port selection.
"""

import socket
import subprocess
import sys
import os
import urllib.request
import urllib.error
from tensorboard.config import get_runs_directory, get_default_port, get_max_port_attempts


def is_port_in_use(port: int) -> bool:
    """
    Check if a port is already in use.
    
    Args:
        port: Port number to check
    
    Returns:
        True if port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def is_tensorboard_running(port: int) -> bool:
    """
    Check if TensorBoard is actually running on the specified port.
    
    Args:
        port: Port number to check
    
    Returns:
        True if TensorBoard is responding on the port, False otherwise
    """
    try:
        # Try to connect to the TensorBoard web interface
        with urllib.request.urlopen(f'http://localhost:{port}/', timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
        return False


def launch_tensorboard_server(log_dir: str, start_port: int = None, max_attempts: int = None) -> None:
    """
    Launch TensorBoard server as a background process.
    
    First checks if TensorBoard is already running on the default port. If so,
    displays a message with the access URL. If not, attempts to start TensorBoard
    on the specified port. If the port is already in use, automatically retries on
    the next port (start_port + 1, start_port + 2, etc.) up to max_attempts times.
    
    Args:
        log_dir: Directory path for TensorBoard logs
        start_port: Initial port to try (default: from config, fallback to 6006)
        max_attempts: Maximum number of port attempts (default: from config, fallback to 10)
    
    Behavior:
        - Checks if TensorBoard is already running on default port
        - If running, prints access URL and exits
        - If not running, launches TensorBoard as a background subprocess
        - If successful, prints clickable HTTP link and runs directory name
        - If all ports fail, prints manual launch command as fallback
        - Handles import errors gracefully (TensorBoard not installed)
    """
    # Load configuration
    if start_port is None:
        start_port = get_default_port()
    if max_attempts is None:
        max_attempts = get_max_port_attempts()
    
    runs_dir = get_runs_directory()
    
    # Check if tensorboard is available
    try:
        import tensorboard
    except ImportError:
        print("\nTensorBoard not installed. To view logs manually, install with:")
        print(f"  uv pip install tensorboard")
        print(f"Then run: tensorboard --logdir={log_dir}")
        return
    
    # Check if TensorBoard is already running on the default port
    if is_tensorboard_running(start_port):
        print(f"\n✓ TensorBoard is already running at http://localhost:{start_port}/")
        print(f"  Your new run logs are at: {log_dir}")
        print(f"  Default runs directory: {runs_dir}")
        print(f"  Click here to open: http://localhost:{start_port}/")
        return
    
    print("\nLaunching TensorBoard server...")
    
    # Try to find an available port
    port = start_port
    for attempt in range(max_attempts):
        if not is_port_in_use(port):
            # Port is available, try to launch TensorBoard
            try:
                # Launch TensorBoard as a background process
                # Use DEVNULL to suppress output and detach from parent process
                subprocess.Popen(
                    [sys.executable, '-m', 'tensorboard.main', '--logdir', log_dir, '--port', str(port), '--host', 'localhost'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True  # Detach from parent process
                )
                
                print(f"✓ TensorBoard launched successfully!")
                print(f"  Click here to open: http://localhost:{port}/")
                print(f"  Viewing logs from: {log_dir}")
                print(f"  Default runs directory: {runs_dir}")
                return
                
            except Exception as e:
                print(f"Failed to launch TensorBoard on port {port}: {e}")
                # Try next port
                port += 1
                continue
        else:
            # Port in use, try next one
            if attempt == 0:
                print(f"Port {port} in use, trying {port + 1}...")
            else:
                print(f"Port {port} in use, trying {port + 1}...")
            port += 1
    
    # All attempts failed, provide manual command
    print(f"\nCould not launch TensorBoard automatically (ports {start_port}-{port-1} unavailable).")
    print(f"To view logs manually, run:")
    print(f"  tensorboard --logdir={log_dir}")
    print(f"  Default runs directory: {runs_dir}")
