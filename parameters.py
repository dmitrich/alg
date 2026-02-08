"""
Parameters module for GPT refactoring project.

This module provides a type-safe container for model hyperparameters using
a dataclass-based approach. It centralizes parameter management and provides
utilities for loading, validation, and backward compatibility.
"""

from dataclasses import dataclass
import json
import torch


@dataclass
class ModelConfig:
    """
    Type-safe container for model hyperparameters.
    
    This dataclass encapsulates all training parameters, model architecture
    parameters, and device configuration for the GPT model. It provides a
    centralized, type-safe way to manage hyperparameters throughout the
    training and inference pipeline.
    
    Attributes:
        Training Parameters:
            batch_size (int): Number of independent sequences to process in parallel
            block_size (int): Maximum context length for predictions
            max_iters (int): Total number of training iterations
            eval_interval (int): Number of iterations between evaluations
            learning_rate (float): Learning rate for the optimizer
            eval_iters (int): Number of iterations to average for evaluation loss
        
        Model Architecture Parameters:
            n_embd (int): Embedding dimension size
            n_head (int): Number of attention heads (must divide n_embd evenly)
            n_layer (int): Number of transformer blocks
            dropout (float): Dropout probability (must be in range [0, 1))
            vocab_size (int): Size of the vocabulary
        
        Device Configuration:
            device (str): Device to use for computation ('cuda' or 'cpu')
                         Defaults to 'cuda' if available, otherwise 'cpu'
    
    Example:
        >>> config = ModelConfig(
        ...     batch_size=4,
        ...     block_size=256,
        ...     max_iters=3000,
        ...     eval_interval=100,
        ...     learning_rate=0.01,
        ...     eval_iters=200,
        ...     n_embd=128,
        ...     n_head=4,
        ...     n_layer=4,
        ...     dropout=0.0,
        ...     vocab_size=65
        ... )
    """
    
    # Training parameters
    batch_size: int
    block_size: int
    max_iters: int
    eval_interval: int
    learning_rate: float
    eval_iters: int
    
    # Model architecture parameters
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    vocab_size: int
    
    # Device configuration with default based on CUDA availability
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def validate(self) -> None:
        """
        Validate all model configuration parameters.
        
        This method performs comprehensive validation of all hyperparameters
        to ensure they meet the required constraints. It should be called
        after creating a ModelConfig instance to catch configuration errors
        early before training or inference begins.
        
        Raises:
            ValueError: If any parameter fails validation with a descriptive
                       message indicating which parameter is invalid and why.
        
        Validation Rules:
            - batch_size must be > 0
            - block_size must be > 0
            - max_iters must be > 0
            - eval_interval must be > 0
            - learning_rate must be > 0
            - eval_iters must be > 0
            - n_embd must be > 0
            - n_head must be > 0
            - n_layer must be > 0
            - dropout must be in range [0, 1)
            - n_embd must be divisible by n_head (for multi-head attention)
        
        Example:
            >>> config = ModelConfig(batch_size=-1, ...)
            >>> config.validate()  # Raises ValueError
            ValueError: batch_size must be greater than 0, got -1
        """
        # Validate training parameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {self.batch_size}")
        
        if self.block_size <= 0:
            raise ValueError(f"block_size must be greater than 0, got {self.block_size}")
        
        if self.max_iters <= 0:
            raise ValueError(f"max_iters must be greater than 0, got {self.max_iters}")
        
        if self.eval_interval <= 0:
            raise ValueError(f"eval_interval must be greater than 0, got {self.eval_interval}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be greater than 0, got {self.learning_rate}")
        
        if self.eval_iters <= 0:
            raise ValueError(f"eval_iters must be greater than 0, got {self.eval_iters}")
        
        # Validate model architecture parameters
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be greater than 0, got {self.n_embd}")
        
        if self.n_head <= 0:
            raise ValueError(f"n_head must be greater than 0, got {self.n_head}")
        
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be greater than 0, got {self.n_layer}")
        
        # Validate dropout is in valid range [0, 1)
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f"dropout must be in range [0, 1), got {self.dropout}")
        
        # Validate n_embd is divisible by n_head (required for multi-head attention)
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd must be divisible by n_head for multi-head attention. "
                f"Got n_embd={self.n_embd}, n_head={self.n_head}, "
                f"remainder={self.n_embd % self.n_head}"
            )
    
    @classmethod
    def from_json(cls, config_path: str, vocab_size: int = None) -> 'ModelConfig':
        """
        Load configuration from JSON file.
        
        This classmethod creates a ModelConfig instance by loading and parsing
        a JSON configuration file. It automatically validates the loaded
        configuration to ensure all parameters meet the required constraints.
        
        Args:
            config_path (str): Path to the JSON configuration file.
            vocab_size (int, optional): Vocabulary size to use if not present in JSON.
                                       If provided, this will override any vocab_size
                                       in the JSON file.
        
        Returns:
            ModelConfig: A validated ModelConfig instance with parameters
                        loaded from the JSON file.
        
        Raises:
            FileNotFoundError: If the specified config file does not exist.
                              The error message includes the path that was attempted.
            json.JSONDecodeError: If the file exists but contains invalid JSON.
                                 The error message includes details about the parsing error.
            TypeError: If vocab_size is not provided and not present in the JSON file.
            ValueError: If the loaded configuration fails validation (e.g., invalid
                       parameter values). Raised by the validate() method.
        
        Example:
            >>> # Load config with vocab_size in JSON
            >>> config = ModelConfig.from_json('config.json')
            >>> print(config.batch_size)
            4
            
            >>> # Load config and provide vocab_size separately
            >>> config = ModelConfig.from_json('config.json', vocab_size=65)
            >>> print(config.vocab_size)
            65
            
            >>> # Handles missing file gracefully
            >>> config = ModelConfig.from_json('missing.json')
            FileNotFoundError: Configuration file not found: missing.json
            
            >>> # Handles invalid JSON gracefully
            >>> config = ModelConfig.from_json('invalid.json')
            json.JSONDecodeError: Invalid JSON in configuration file 'invalid.json': ...
        
        Note:
            The JSON file must contain all required fields except 'device' and 'vocab_size'.
            The 'device' field defaults to 'cuda' if available, otherwise 'cpu'.
            The 'vocab_size' can be provided either in the JSON file or as a parameter.
        """
        # Handle FileNotFoundError with clear error message
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file '{config_path}': {e.msg}",
                e.doc,
                e.pos
            )
        
        # If vocab_size is provided as parameter, use it (override JSON if present)
        if vocab_size is not None:
            config_dict['vocab_size'] = vocab_size
        
        # Create ModelConfig instance from parsed JSON
        config = cls(**config_dict)
        
        # Call validate() after loading
        config.validate()
        
        # Return ModelConfig instance
        return config
    
    @classmethod
    def from_dict(cls, config_dict: dict, vocab_size: int) -> 'ModelConfig':
        """
        Create ModelConfig instance from dictionary.
        
        This classmethod creates a ModelConfig instance by merging a configuration
        dictionary with a vocab_size parameter. This is useful when loading
        configuration from sources that don't include vocab_size (which is typically
        determined from the training data).
        
        Args:
            config_dict (dict): Dictionary containing configuration parameters.
                               Should include all required fields except vocab_size.
            vocab_size (int): Size of the vocabulary (determined from training data).
        
        Returns:
            ModelConfig: A ModelConfig instance with parameters from the dictionary
                        and the provided vocab_size.
        
        Example:
            >>> config_dict = {
            ...     'batch_size': 4,
            ...     'block_size': 256,
            ...     'max_iters': 3000,
            ...     'eval_interval': 100,
            ...     'learning_rate': 0.01,
            ...     'eval_iters': 200,
            ...     'n_embd': 128,
            ...     'n_head': 4,
            ...     'n_layer': 4,
            ...     'dropout': 0.0
            ... }
            >>> config = ModelConfig.from_dict(config_dict, vocab_size=65)
            >>> print(config.vocab_size)
            65
        
        Note:
            This method does not call validate() automatically. The caller should
            call validate() on the returned instance if validation is needed.
        """
        # Merge config dict with vocab_size
        merged_dict = {**config_dict, 'vocab_size': vocab_size}
        
        # Create ModelConfig instance
        return cls(**merged_dict)
    
    def to_dict(self) -> dict:
        """
        Convert ModelConfig to dictionary.
        
        This method converts the ModelConfig dataclass instance to a dictionary
        representation. This is useful for serialization, logging, or passing
        configuration to other components.
        
        Returns:
            dict: Dictionary representation of the ModelConfig with all fields
                 as key-value pairs.
        
        Example:
            >>> config = ModelConfig(
            ...     batch_size=4,
            ...     block_size=256,
            ...     max_iters=3000,
            ...     eval_interval=100,
            ...     learning_rate=0.01,
            ...     eval_iters=200,
            ...     n_embd=128,
            ...     n_head=4,
            ...     n_layer=4,
            ...     dropout=0.0,
            ...     vocab_size=65
            ... )
            >>> config_dict = config.to_dict()
            >>> print(config_dict['batch_size'])
            4
            >>> print(config_dict['vocab_size'])
            65
        """
        # Convert dataclass to dictionary using dataclasses.asdict()
        from dataclasses import asdict
        return asdict(self)
    
    def apply_to_model_module(self, model_module) -> None:
        """
        Set global variables in model module from config (for backward compatibility).
        
        This method provides backward compatibility with the existing model.py that
        uses global variables for hyperparameters. It sets all the global variables
        in the provided model module to match the values in this ModelConfig instance.
        
        This is a temporary compatibility layer to allow gradual migration from
        global variables to the ModelConfig approach. New code should use the
        ModelConfig instance directly rather than relying on global variables.
        
        Args:
            model_module: The model module (typically imported as 'import model')
                         whose global variables should be updated.
        
        Returns:
            None: This method modifies the module in-place and returns nothing.
        
        Example:
            >>> import model
            >>> config = ModelConfig.from_json('config.json', vocab_size=65)
            >>> config.apply_to_model_module(model)
            >>> # Now model.batch_size, model.block_size, etc. are set from config
            >>> print(model.batch_size)
            4
            >>> print(model.n_embd)
            128
        
        Note:
            This method sets the following global variables in the model module:
            - batch_size: Number of sequences processed in parallel
            - block_size: Maximum context length for predictions
            - max_iters: Total number of training iterations
            - eval_interval: Number of iterations between evaluations
            - learning_rate: Learning rate for the optimizer
            - device: Device to use for computation ('cuda' or 'cpu')
            - eval_iters: Number of iterations to average for evaluation
            - n_embd: Embedding dimension size
            - n_head: Number of attention heads
            - n_layer: Number of transformer blocks
            - dropout: Dropout probability
            - vocab_size: Size of the vocabulary
        """
        # Set all global variables in model module from config
        model_module.batch_size = self.batch_size
        model_module.block_size = self.block_size
        model_module.max_iters = self.max_iters
        model_module.eval_interval = self.eval_interval
        model_module.learning_rate = self.learning_rate
        model_module.device = self.device
        model_module.eval_iters = self.eval_iters
        model_module.n_embd = self.n_embd
        model_module.n_head = self.n_head
        model_module.n_layer = self.n_layer
        model_module.dropout = self.dropout
        model_module.vocab_size = self.vocab_size
