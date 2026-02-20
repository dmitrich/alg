"""
TensorBoard writer management for training visualization.

This module provides the TensorBoardWriter class that manages the lifecycle
of PyTorch's SummaryWriter with graceful error handling and user-friendly
output messages.
"""

import os
import shutil

class TensorBoardWriter:
    """Manages TensorBoard SummaryWriter lifecycle with dual-location logging."""
    
    def __init__(self, log_dir: str, secondary_log_dir: str = None):
        """
        Initialize TensorBoard writer.
        
        Args:
            log_dir: Primary directory path for TensorBoard logs
            secondary_log_dir: Optional secondary directory to copy logs to
        """
        self.log_dir = log_dir
        self.secondary_log_dir = secondary_log_dir
        self.writer = None
        self.available = False
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize SummaryWriter with error handling."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.available = True
        except ImportError:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
        except Exception as e:
            print(f"Warning: TensorBoard initialization failed: {e}")
    
    def close(self) -> None:
        """Close writer, copy logs to secondary location, and print access information."""
        if self.available and self.writer is not None:
            self.writer.close()
            
            if self.secondary_log_dir:
                try:
                    os.makedirs(self.secondary_log_dir, exist_ok=True)
                    
                    for filename in os.listdir(self.log_dir):
                        if filename.startswith('events.out.tfevents'):
                            src = os.path.join(self.log_dir, filename)
                            dst = os.path.join(self.secondary_log_dir, filename)
                            shutil.copy2(src, dst)
                    
                    print(f"\n✓ TensorBoard logs saved to:")
                    print(f"  Primary: {self.log_dir}")
                    print(f"  Secondary: {self.secondary_log_dir}")
                except Exception as e:
                    print(f"\n✓ TensorBoard logs saved to: {self.log_dir}")
                    print(f"  Warning: Failed to copy to secondary location: {e}")
            else:
                print(f"\n✓ TensorBoard logs saved to: {self.log_dir}")
    
    def is_available(self) -> bool:
        """Check if TensorBoard is available."""
        return self.available
