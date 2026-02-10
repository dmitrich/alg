"""
Console output formatting utilities for ALG1 training project.
Handles clean, compact summary output for training and inference.
"""

from datetime import datetime


def print_run_summary(run_id: str, config: dict, start_time: str) -> None:
    """
    Print clean, compact run configuration summary.
    
    Args:
        run_id: Run identifier
        config: Merged configuration dictionary
        start_time: ISO format timestamp
        
    Output format:
        Run: {run_id}
        Started: {start_time}
        Config: {key config params on one line}
    """
    # Format config as compact one-liner
    config_str = f"lr={config.get('learning_rate', 'N/A')} batch={config.get('batch_size', 'N/A')} iters={config.get('max_iters', 'N/A')} embd={config.get('n_embd', 'N/A')} layers={config.get('n_layer', 'N/A')}"
    
    print(f"Run: {run_id}")
    print(f"Started: {start_time}")
    print(f"Config: {config_str}")
    print()


def print_completion_summary(run_id: str, config: dict, start_time: str, end_time: str) -> None:
    """
    Print completion summary with total run time.
    
    Args:
        run_id: Run identifier
        config: Merged configuration dictionary
        start_time: ISO format timestamp
        end_time: ISO format timestamp
        
    Output format:
        Run: {run_id}
        Completed: {end_time}
        Duration: {elapsed time}
        Config: {key config params on one line}
    """
    # Calculate duration
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    duration = end_dt - start_dt
    
    # Format duration as human-readable
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"
    
    # Format config as compact one-liner
    config_str = f"lr={config.get('learning_rate', 'N/A')} batch={config.get('batch_size', 'N/A')} iters={config.get('max_iters', 'N/A')} embd={config.get('n_embd', 'N/A')} layers={config.get('n_layer', 'N/A')}"
    
    print(f"\nRun: {run_id}")
    print(f"Completed: {end_time}")
    print(f"Duration: {duration_str}")
    print(f"Config: {config_str}")
    print()


def print_inference_summary(run_id: str, config: dict, start_time: str) -> None:
    """
    Print clean, compact inference configuration summary.
    
    Args:
        run_id: Run identifier (or "root" for backward compat)
        config: Configuration dictionary
        start_time: ISO format timestamp
        
    Output format:
        Model: {run_id or path}
        Started: {start_time}
        Config: {key config params on one line}
    """
    # Format config as compact one-liner
    config_str = f"embd={config.get('n_embd', 'N/A')} layers={config.get('n_layer', 'N/A')} heads={config.get('n_head', 'N/A')} block={config.get('block_size', 'N/A')}"
    
    print(f"Model: {run_id}")
    print(f"Started: {start_time}")
    print(f"Config: {config_str}")
    print()


def print_inference_completion(run_id: str, config: dict, start_time: str, end_time: str) -> None:
    """
    Print completion summary with total inference time.
    
    Args:
        run_id: Run identifier (or "root" for backward compat)
        config: Configuration dictionary
        start_time: ISO format timestamp
        end_time: ISO format timestamp
        
    Output format:
        Model: {run_id or path}
        Completed: {end_time}
        Duration: {elapsed time}
        Config: {key config params on one line}
    """
    # Calculate duration
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    duration = end_dt - start_dt
    
    # Format duration as human-readable
    total_seconds = duration.total_seconds()
    if total_seconds < 1:
        duration_str = f"{total_seconds*1000:.0f}ms"
    elif total_seconds < 60:
        duration_str = f"{total_seconds:.1f}s"
    else:
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        duration_str = f"{minutes}m {seconds:.1f}s"
    
    # Format config as compact one-liner
    config_str = f"embd={config.get('n_embd', 'N/A')} layers={config.get('n_layer', 'N/A')} heads={config.get('n_head', 'N/A')} block={config.get('block_size', 'N/A')}"
    
    print(f"\nModel: {run_id}")
    print(f"Completed: {end_time}")
    print(f"Duration: {duration_str}")
    print(f"Config: {config_str}")
    print()
