"""
AimTracker: Wrapper for Aim experiment tracking integration.

This module provides a fail-safe wrapper around the Aim SDK that handles
initialization, metric tracking, hyperparameter logging, and cleanup with
comprehensive error handling to ensure training continues even if Aim fails.
"""

from typing import Dict, Any, Optional
import numpy as np


class AimTracker:
    """
    Wrapper for Aim experiment tracking that integrates with existing training pipeline.
    
    This class provides a simple interface for tracking metrics, hyperparameters,
    and model distributions while handling initialization and cleanup automatically.
    All operations are wrapped in try-except blocks to ensure training continues
    even if Aim encounters errors.
    """
    
    def __init__(self, run_name: str, run_dir: str, repo_path: str = '.aim'):
        """
        Initialize Aim tracking for a training run.
        
        Args:
            run_name: Unique identifier for this run (e.g., "2026-02-10_007")
            run_dir: Path to the run directory containing artifacts and logs
            repo_path: Path to Aim repository (default: '.aim')
        """
        try:
            from aim import Run
            
            self.run = Run(
                run_hash=None,  # Let Aim generate unique hash
                repo=repo_path,
                experiment=run_name  # Use run_name as experiment name
            )
            
            # Store run directory path in Aim Run metadata
            self.run['run_dir'] = run_dir
            self.run['run_name'] = run_name
            self.enabled = True
            
        except Exception as e:
            print(f"Warning: Failed to initialize Aim tracking: {e}")
            print("Training will continue without Aim tracking.")
            self.run = None
            self.enabled = False
    
    def track_config(self, config: Dict[str, Any]):
        """
        Track all hyperparameters and configuration values.
        
        Args:
            config: Dictionary containing all hyperparameters
        """
        if not self.enabled:
            return
        
        try:
            # Store entire config dict in Aim Run
            self.run['hparams'] = config
            
            # Also store individual params for easier filtering in UI
            for key, value in config.items():
                self.run[key] = value
                
        except Exception as e:
            print(f"Warning: Failed to track config: {e}")
    
    def track_metric(self, name: str, value: float, step: int, context: Optional[Dict[str, str]] = None):
        """
        Track a single metric value.
        
        Args:
            name: Metric name (e.g., 'loss', 'accuracy')
            value: Metric value
            step: Training iteration/step number
            context: Optional context dict for grouping (e.g., {'subset': 'train'})
        """
        if not self.enabled:
            return
        
        try:
            self.run.track(value, name=name, step=step, context=context or {})
        except Exception as e:
            print(f"Warning: Failed to track metric {name}: {e}")
    
    def track_distribution(self, name: str, values: np.ndarray, step: int, context: Optional[Dict[str, str]] = None):
        """
        Track parameter or gradient distribution.
        
        Args:
            name: Distribution name (e.g., 'layer1.weight')
            values: Numpy array of parameter/gradient values
            step: Training iteration/step number
            context: Optional context dict for grouping
        """
        if not self.enabled:
            return
        
        try:
            from aim import Distribution
            self.run.track(Distribution(values), name=name, step=step, context=context or {})
        except Exception as e:
            print(f"Warning: Failed to track distribution {name}: {e}")
    
    def track_metadata(self, key: str, value: Any):
        """
        Track arbitrary metadata using dictionary-like interface.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        if not self.enabled:
            return
        
        try:
            self.run[key] = value
        except Exception as e:
            print(f"Warning: Failed to track metadata {key}: {e}")
    
    def close(self):
        """
        Finalize the Aim Run and flush all data.
        Should be called when training completes or fails.
        """
        if not self.enabled:
            return
        
        try:
            if self.run:
                self.run.close()
        except Exception as e:
            print(f"Warning: Failed to close Aim run: {e}")
