"""
Training script for GPT model.

This script trains a character-level GPT (Generative Pre-trained Transformer) model
on text data (Shakespeare dataset by default). The training process:
1. Loads configuration and data
2. Creates a character-level tokenizer
3. Initializes the GPT model
4. Trains using AdamW optimizer with cross-entropy loss
5. Saves trained weights to disk

Usage:
    python train.py                          # Use default config.json
    python train.py --config my_config.json  # Use custom config
    python train.py --max-iters 10000        # Override specific parameters
"""

# Added for modular imports - PyTorch core modules
import torch

# Added for CLI support - command-line argument parsing
import argparse

# Added for modular imports - configuration management
from config import load_config, validate_config

# Added for modular imports - parameters module for ModelConfig
from parameters import ModelConfig

# Added for modular imports - data loading and batch generation
from data import load_data, get_batch

# Added for modular imports - tokenization utilities
from utils import create_tokenizer

# Added for modular imports - model architecture
import model
from model import BigramLanguageModel

# Added for new folder structure - configuration and run management
from utils_alg1.utils_config_loader import ConfigLoader
from utils_alg1.utils_run_manager import RunIDGenerator, RunDirectoryCreator, MetadataCapture
from utils_alg1.utils_output import print_run_summary, print_completion_summary
from datetime import datetime
import os

# Added for TensorBoard integration - logging and visualization
from utils_tensorboard.writer import TensorBoardWriter
from utils_tensorboard.logger import MetricLogger
from utils_tensorboard.instructions import print_tensorboard_instructions
from utils_tensorboard.config import get_histogram_interval

# Added for Aim experiment tracking - comprehensive experiment tracking and visualization
from utils_aim.tracker import AimTracker
from utils_aim.instructions import print_aim_instructions


# Added for CLI support - parse command-line arguments for hyperparameter overrides
parser = argparse.ArgumentParser(description='Train GPT model on text data')
parser.add_argument('--config', type=str, default='config.json',
                    help='Path to configuration file (default: config.json)')
parser.add_argument('--data', type=str, default='data/input.txt',
                    help='Path to training data file (default: data/input.txt)')
parser.add_argument('--output', type=str, default='model.safetensors',
                    help='Path to save trained model in SafeTensors format (default: model.safetensors)')
parser.add_argument('--batch-size', type=int, default=None,
                    help='Batch size for training (overrides config.json)')
parser.add_argument('--block-size', type=int, default=None,
                    help='Context length / block size (overrides config.json)')
parser.add_argument('--max-iters', type=int, default=None,
                    help='Maximum training iterations (overrides config.json)')
parser.add_argument('--learning-rate', type=float, default=None,
                    help='Learning rate for optimizer (overrides config.json)')
parser.add_argument('--eval-interval', type=int, default=None,
                    help='Evaluate loss every N iterations (overrides config.json)')
parser.add_argument('--eval-iters', type=int, default=None,
                    help='Number of iterations for loss estimation (overrides config.json)')
parser.add_argument('--n-embd', type=int, default=None,
                    help='Embedding dimension (overrides config.json)')
parser.add_argument('--n-head', type=int, default=None,
                    help='Number of attention heads (overrides config.json)')
parser.add_argument('--n-layer', type=int, default=None,
                    help='Number of transformer layers (overrides config.json)')
parser.add_argument('--dropout', type=float, default=None,
                    help='Dropout rate (overrides config.json)')
parser.add_argument('--run-tag', type=str, default=None,
                    help='Optional tag to append to run ID (e.g., "baseline", "high-lr")')
args = parser.parse_args()


def setup_training_run(args):
    """
    Set up training run with new directory structure.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (run_dir, merged_config)
    """
    # Load and merge configurations
    config_loader = ConfigLoader()
    
    # Check if new configs/ directory exists, otherwise fall back to legacy config.json
    if os.path.exists('configs'):
        merged_config = config_loader.load_configs('configs')
    else:
        # Backward compatibility: load from config.json
        merged_config = load_config(args.config)
    
    # Override with CLI arguments
    if args.batch_size is not None:
        merged_config['batch_size'] = args.batch_size
    if args.block_size is not None:
        merged_config['block_size'] = args.block_size
    if args.max_iters is not None:
        merged_config['max_iters'] = args.max_iters
    if args.learning_rate is not None:
        merged_config['learning_rate'] = args.learning_rate
    if args.eval_interval is not None:
        merged_config['eval_interval'] = args.eval_interval
    if args.eval_iters is not None:
        merged_config['eval_iters'] = args.eval_iters
    if args.n_embd is not None:
        merged_config['n_embd'] = args.n_embd
    if args.n_head is not None:
        merged_config['n_head'] = args.n_head
    if args.n_layer is not None:
        merged_config['n_layer'] = args.n_layer
    if args.dropout is not None:
        merged_config['dropout'] = args.dropout
    
    # Generate unique Run_ID
    run_id_gen = RunIDGenerator()
    run_id = run_id_gen.generate(tag=args.run_tag)
    
    # Create run directory structure
    run_dir_creator = RunDirectoryCreator()
    run_dir = run_dir_creator.create(run_id)
    
    # Capture metadata
    metadata_capture = MetadataCapture()
    meta_dir = os.path.join(run_dir, 'meta')
    
    # Copy config files if they exist
    if os.path.exists('configs'):
        metadata_capture.copy_configs('configs', meta_dir)
    
    # Capture git and environment info
    metadata_capture.capture_git_info(os.path.join(meta_dir, 'git.txt'))
    metadata_capture.capture_env_info(os.path.join(meta_dir, 'env.txt'))
    
    # Save full merged config (including data params) to meta directory
    import json
    with open(os.path.join(meta_dir, 'full_config.json'), 'w') as f:
        json.dump(merged_config, f, indent=2)
    
    return run_dir, merged_config, run_id


def get_output_paths(run_dir: str) -> dict:
    """
    Get output paths for artifacts.
    
    Args:
        run_dir: Run directory path
        
    Returns:
        Dictionary of output paths (checkpoints, export, logs)
    """
    return {
        'checkpoints': os.path.join(run_dir, 'artifacts', 'checkpoints'),
        'export': os.path.join(run_dir, 'artifacts', 'export'),
        'logs': os.path.join(run_dir, 'logs')
    }

# Added for configuration loading - load hyperparameters from config.json
# This loads all model and training hyperparameters from the JSON file
# Set up training run with new directory structure
start_time = datetime.now().isoformat()
run_dir, config, run_id = setup_training_run(args)
output_paths = get_output_paths(run_dir)

# Initialize Aim tracker for experiment tracking
aim_tracker = AimTracker(run_id, run_dir)

# Track hyperparameters and configuration in Aim
aim_tracker.track_config(config)

# Track git and environment metadata for reproducibility
metadata_capture = MetadataCapture()
aim_tracker.track_metadata('git_commit', metadata_capture.get_git_commit())
aim_tracker.track_metadata('git_branch', metadata_capture.get_git_branch())
aim_tracker.track_metadata('python_version', metadata_capture.get_python_version())

# Initialize TensorBoard writer for training visualization
tensorboard_dir = os.path.join(run_dir, 'logs', 'tensorboard')
tb_writer = TensorBoardWriter(tensorboard_dir)
tb_logger = MetricLogger(tb_writer)

# Get histogram logging interval from config
histogram_interval = get_histogram_interval()

# Print run summary at start
print_run_summary(run_id, config, start_time)

# Validate that all parameters are present and have valid values
validate_config(config)

# Added for data loading - load and split training data
# This reads the text file and splits it into training (90%) and validation (10%) sets
# Use data_path from config if available, otherwise use CLI argument
data_path = config.get('data_path', args.data)
train_data, val_data, text = load_data(data_path)
print(f"Loaded {len(text)} characters of training data")

# Added for tokenizer creation - create character-level tokenizer
# The tokenizer converts text to integers (encode) and integers back to text (decode)
# vocab_size is the number of unique characters in the dataset
encode, decode, vocab_size = create_tokenizer(text)
print(f"Vocabulary size: {vocab_size}")

# Added for configuration - create ModelConfig instance from loaded config
# This provides type-safe parameter management and validation
# Filter out data-specific parameters that ModelConfig doesn't need
model_config_params = {k: v for k, v in config.items() 
                       if k not in ['data_path', 'train_split', 'val_split']}
config_obj = ModelConfig.from_dict(model_config_params, vocab_size)
# Validate all parameters to catch configuration errors early
config_obj.validate()
# Apply configuration to model module for backward compatibility
config_obj.apply_to_model_module(model)

# Added for configuration loading - extract hyperparameters from config
# These variables control the training process and model architecture
batch_size = config_obj.batch_size        # Number of sequences processed in parallel
block_size = config_obj.block_size        # Context length (number of tokens per sequence)
max_iters = config_obj.max_iters          # Total number of training iterations
eval_interval = config_obj.eval_interval  # How often to evaluate loss
learning_rate = config_obj.learning_rate  # Step size for optimizer
eval_iters = config_obj.eval_iters        # Number of batches to average for loss estimation
n_embd = config_obj.n_embd                # Embedding dimension
n_head = config_obj.n_head                # Number of attention heads
n_layer = config_obj.n_layer              # Number of transformer layers
dropout = config_obj.dropout              # Dropout rate for regularization

# Added for device selection - automatically use GPU if available
# Training on GPU (CUDA) is much faster than CPU for neural networks
device = config_obj.device
print(f"Using device: {device}")

# Added for model initialization - create model instance
# BigramLanguageModel is the GPT model with multi-head attention and transformer blocks
model_instance = BigramLanguageModel()
# Move model to GPU/CPU device
m = model_instance.to(device)
# Print the number of trainable parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

@torch.no_grad()
def estimate_loss() -> dict:
    """
    Estimate average loss on training and validation sets.
    
    This function evaluates the model on multiple batches to get a stable
    estimate of the loss, which is less noisy than single-batch evaluation.
    The @torch.no_grad() decorator disables gradient computation for efficiency.
    """
    out = {}
    # Set model to evaluation mode (disables dropout)
    model_instance.eval()
    
    # Evaluate on both training and validation splits
    for split in ['train', 'val']:
        # Collect losses from multiple batches for stable estimate
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # Get a batch of data from the specified split
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            # Forward pass: compute predictions and loss
            logits, loss = model_instance(X, Y)
            # Store the loss value (convert from tensor to Python float)
            losses[k] = loss.item()
        # Average the losses across all evaluation batches
        out[split] = losses.mean()
    
    # Set model back to training mode (enables dropout)
    model_instance.train()
    return out

# Create PyTorch optimizer - AdamW is Adam with weight decay regularization
# The optimizer will update model parameters to minimize the loss
optimizer = torch.optim.AdamW(model_instance.parameters(), lr=learning_rate)

# Wrap training in try-finally to ensure Aim cleanup happens even on failure
try:
    # Main training loop - iterate for the specified number of training steps
    for iter in range(max_iters):

        # Periodically evaluate loss on train and val sets to monitor training progress
        # This helps us see if the model is learning and if it's overfitting
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Track evaluation metrics in Aim with context for train/val subsets
            aim_tracker.track_metric('loss', losses['train'].item(), iter, context={'subset': 'train'})
            aim_tracker.track_metric('loss', losses['val'].item(), iter, context={'subset': 'val'})
            
            # Log evaluation metrics to TensorBoard
            tb_logger.log_evaluation_metrics(
                losses['train'].item(),
                losses['val'].item(),
                learning_rate,
                iter
            )
        
        # Periodically log model parameter and gradient histograms
        # Only log if histogram_interval is positive
        if histogram_interval > 0 and (iter % histogram_interval == 0 or iter == max_iters - 1):
            tb_logger.log_model_histograms(model_instance, iter)
            
            # Track parameter and gradient distributions in Aim
            for name, param in model_instance.named_parameters():
                aim_tracker.track_distribution(name, param.detach().cpu().numpy(), iter, context={'type': 'param'})
                
                # Track gradient distributions if gradients exist
                if param.grad is not None:
                    aim_tracker.track_distribution(name + '.grad', param.grad.detach().cpu().numpy(), iter, context={'type': 'grad'})

        # Sample a batch of training data
        # xb: input sequences (batch_size, block_size)
        # yb: target sequences (batch_size, block_size) - shifted by 1 position
        xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device)

        # Forward pass: compute model predictions and loss
        # logits: raw prediction scores for each token in vocabulary
        # loss: cross-entropy loss measuring prediction error
        logits, loss = model_instance(xb, yb)
        
        # Backward pass: compute gradients
        # First, zero out gradients from previous iteration
        optimizer.zero_grad(set_to_none=True)
        # Compute gradients of loss with respect to all model parameters
        loss.backward()
        # Update model parameters using computed gradients
        optimizer.step()
        
        # Track training loss in Aim for experiment tracking
        aim_tracker.track_metric('loss', loss.item(), iter, context={'subset': 'train'})
        
        # Log training loss to TensorBoard
        tb_logger.log_training_loss(loss.item(), iter)

finally:
    # Ensure Aim tracker is closed even if training fails
    aim_tracker.close()

# Close TensorBoard writer and flush all data to disk
tb_writer.close()

# Print TensorBoard launch instructions
print_tensorboard_instructions(tensorboard_dir)

# Added for model persistence - save trained model weights after training completes
print("\nTraining complete! Saving model weights...")
end_time = datetime.now().isoformat()

# Save model to run-specific export directory
model_path = os.path.join(output_paths['export'], 'model.safetensors')
config_path = os.path.join(output_paths['export'], 'config.json')

model_instance.save_safetensors(model_path)
print(f"Model weights saved to '{model_path}' in SafeTensors format")

# Save config alongside model
import json
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuration saved to '{config_path}'")

# Print Aim UI launch instructions
print_aim_instructions()

# Print completion summary
print_completion_summary(run_id, config, start_time, end_time)
