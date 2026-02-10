"""
Metric logging utilities for TensorBoard integration.

This module provides the MetricLogger class that offers high-level methods
for logging different types of training metrics to TensorBoard with graceful
error handling.
"""

from utils_tensorboard.writer import TensorBoardWriter


class MetricLogger:
    """Logs training metrics to TensorBoard."""
    
    def __init__(self, writer: TensorBoardWriter):
        """
        Initialize metric logger.
        
        Args:
            writer: TensorBoardWriter instance
        """
        self.writer = writer
    
    def log_training_loss(self, loss: float, step: int) -> None:
        """
        Log training loss for current iteration.
        
        Args:
            loss: Training loss value
            step: Current iteration/step number
        """
        if self.writer.is_available():
            try:
                self.writer.writer.add_scalar('Loss/train_step', loss, step)
            except Exception as e:
                print(f"Warning: Failed to log training loss: {e}")
                self.writer.available = False
    
    def log_evaluation_metrics(self, train_loss: float, val_loss: float, 
                               learning_rate: float, step: int) -> None:
        """
        Log evaluation metrics.
        
        Args:
            train_loss: Average training loss over evaluation iterations
            val_loss: Average validation loss over evaluation iterations
            learning_rate: Current learning rate
            step: Current iteration/step number
        """
        if self.writer.is_available():
            try:
                self.writer.writer.add_scalar('Loss/train', train_loss, step)
                self.writer.writer.add_scalar('Loss/val', val_loss, step)
                self.writer.writer.add_scalar('Learning_Rate', learning_rate, step)
            except Exception as e:
                print(f"Warning: Failed to log evaluation metrics: {e}")
                self.writer.available = False
    
    def log_gradient_norm(self, grad_norm: float, step: int) -> None:
        """
        Log gradient norm (when gradient clipping is enabled).
        
        Args:
            grad_norm: Gradient norm value
            step: Current iteration/step number
        """
        if self.writer.is_available():
            try:
                self.writer.writer.add_scalar('Gradients/norm', grad_norm, step)
            except Exception as e:
                print(f"Warning: Failed to log gradient norm: {e}")
                self.writer.available = False
    
    def log_model_histograms(self, model, step: int) -> None:
        """
        Log histograms of model parameters and gradients.
        
        Args:
            model: PyTorch model instance
            step: Current iteration/step number
        """
        if self.writer.is_available():
            try:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # Log parameter values
                        self.writer.writer.add_histogram(
                            f'Parameters/{name}', 
                            param.data.cpu(), 
                            step
                        )
                        # Log gradients if they exist
                        if param.grad is not None:
                            self.writer.writer.add_histogram(
                                f'Gradients/{name}', 
                                param.grad.data.cpu(), 
                                step
                            )
            except Exception as e:
                print(f"Warning: Failed to log model histograms: {e}")
                self.writer.available = False
    
    def log_activation_histograms(self, activations: dict, step: int) -> None:
        """
        Log histograms of layer activations.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors
            step: Current iteration/step number
        """
        if self.writer.is_available():
            try:
                for name, activation in activations.items():
                    self.writer.writer.add_histogram(
                        f'Activations/{name}', 
                        activation.data.cpu(), 
                        step
                    )
            except Exception as e:
                print(f"Warning: Failed to log activation histograms: {e}")
                self.writer.available = False
