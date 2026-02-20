"""
Aim UI launch instructions printer.

This module provides functionality to print instructions for manually
launching the Aim UI to view and compare experiment runs.
"""

def print_aim_instructions(repo_path: str = '.aim') -> None:
    """
    Print instructions for launching Aim UI manually.
    
    Args:
        repo_path: Path to Aim repository (default: '.aim')
    """
    print("\n" + "="*70)
    print("Aim Experiment Tracking")
    print("="*70)
    print(f"\nYour experiment data has been tracked in the Aim repository:")
    print(f"  {repo_path}")
    print(f"\nTo view and compare your experiments in the Aim UI, run:")
    print(f"\n  aim up")
    print(f"\nThen open your browser to: http://localhost:43800/")
    print(f"\nIn the Aim UI, you can:")
    print(f"  - View all tracked runs with their metrics")
    print(f"  - Filter and compare runs by hyperparameters")
    print(f"  - See metric plots over training iterations")
    print(f"  - Explore parameter and gradient distributions")
    print("="*70 + "\n")
