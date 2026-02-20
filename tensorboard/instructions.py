"""
TensorBoard launch instructions printer.

This module provides functionality to print instructions for manually
launching TensorBoard with the correct configuration.
"""

from tensorboard.config import get_runs_directory, get_default_port

def print_tensorboard_instructions(log_dir: str) -> None:
    """
    Print instructions for launching TensorBoard manually.
    
    Args:
        log_dir: Directory path for TensorBoard logs
    """
    runs_dir = get_runs_directory()
    default_port = get_default_port()
    
    print("\n" + "="*70)
    print("TensorBoard Visualization")
    print("="*70)
    print(f"\nYour training logs have been saved to:")
    print(f"  {log_dir}")
    print(f"\nOr to view all runs in the '{runs_dir}' directory:")
    print(f"\n  tensorboard --logdir={runs_dir} --port={default_port}")
    print(f"\nThen open your browser to: http://localhost:{default_port}/")
    print(f"\nConfiguration: configs/tensorboard.json")
    print(f"  - Default runs directory: {runs_dir}")
    print(f"  - Default port: {default_port}")
    print("="*70 + "\n")
