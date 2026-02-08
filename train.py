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
args = parser.parse_args()

# Added for configuration loading - load hyperparameters from config.json
# This loads all model and training hyperparameters from the JSON file
config = load_config(args.config)
# Validate that all parameters are present and have valid values
validate_config(config)

# Added for CLI support - override config values with command-line arguments if provided
# This allows users to experiment with different hyperparameters without editing config.json
if args.batch_size is not None:
    config['batch_size'] = args.batch_size
if args.block_size is not None:
    config['block_size'] = args.block_size
if args.max_iters is not None:
    config['max_iters'] = args.max_iters
if args.learning_rate is not None:
    config['learning_rate'] = args.learning_rate
if args.eval_interval is not None:
    config['eval_interval'] = args.eval_interval
if args.eval_iters is not None:
    config['eval_iters'] = args.eval_iters
if args.n_embd is not None:
    config['n_embd'] = args.n_embd
if args.n_head is not None:
    config['n_head'] = args.n_head
if args.n_layer is not None:
    config['n_layer'] = args.n_layer
if args.dropout is not None:
    config['dropout'] = args.dropout

# Added for data loading - load and split training data
# This reads the text file and splits it into training (90%) and validation (10%) sets
train_data, val_data, text = load_data(args.data)
print(f"Loaded {len(text)} characters of training data")

# Added for tokenizer creation - create character-level tokenizer
# The tokenizer converts text to integers (encode) and integers back to text (decode)
# vocab_size is the number of unique characters in the dataset
encode, decode, vocab_size = create_tokenizer(text)
print(f"Vocabulary size: {vocab_size}")

# Added for configuration - create ModelConfig instance from loaded config
# This provides type-safe parameter management and validation
config_obj = ModelConfig.from_dict(config, vocab_size)
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

# Main training loop - iterate for the specified number of training steps
for iter in range(max_iters):

    # Periodically evaluate loss on train and val sets to monitor training progress
    # This helps us see if the model is learning and if it's overfitting
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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

# Added for model persistence - save trained model weights after training completes
print("\nTraining complete! Saving model weights...")
model_instance.save_safetensors(args.output)
print(f"Model weights saved to '{args.output}' in SafeTensors format")
