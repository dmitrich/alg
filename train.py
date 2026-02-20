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

import torch

import argparse

from config import load_config, validate_config

from parameters import ModelConfig

from data import load_data, get_batch

from tokenizer import create_tokenizer

import model
from model import LanguageModel

from utils.alg.utils_config_loader import ConfigLoader
from utils.alg.utils_run_manager import RunIDGenerator, RunDirectoryCreator, MetadataCapture
from utils.alg.utils_output import print_run_summary, print_completion_summary
from datetime import datetime
import os

from tensorboard.writer import TensorBoardWriter
from tensorboard.logger import MetricLogger
from tensorboard.instructions import print_tensorboard_instructions
from tensorboard.config import get_histogram_interval

from utils.aim.tracker import AimTracker
from utils.aim.instructions import print_aim_instructions

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
    config_loader = ConfigLoader()
    
    if os.path.exists('configs'):
        merged_config = config_loader.load_configs('configs')
    else:
        merged_config = load_config(args.config)
    
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
    
    run_id_gen = RunIDGenerator()
    run_id = run_id_gen.generate(tag=args.run_tag)
    
    run_dir_creator = RunDirectoryCreator()
    run_dir = run_dir_creator.create(run_id)
    
    metadata_capture = MetadataCapture()
    meta_dir = os.path.join(run_dir, 'meta')
    
    if os.path.exists('configs'):
        metadata_capture.copy_configs('configs', meta_dir)
    
    metadata_capture.capture_git_info(os.path.join(meta_dir, 'git.txt'))
    metadata_capture.capture_env_info(os.path.join(meta_dir, 'env.txt'))
    
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

start_time = datetime.now().isoformat()
run_dir, config, run_id = setup_training_run(args)
output_paths = get_output_paths(run_dir)

aim_enabled = config.get('aim_logging', False)
if aim_enabled:
    aim_tracker = AimTracker(run_id, run_dir)
    
    aim_tracker.track_config(config)
    
    metadata_capture = MetadataCapture()
    aim_tracker.track_metadata('git_commit', metadata_capture.get_git_commit())
    aim_tracker.track_metadata('git_branch', metadata_capture.get_git_branch())
    aim_tracker.track_metadata('python_version', metadata_capture.get_python_version())
else:
    aim_tracker = None
    print("Aim logging disabled (aim_logging=false in config)")

tensorboard_dir = os.path.join(run_dir, 'logs', 'tensorboard')
tb_writer = TensorBoardWriter(tensorboard_dir)
tb_logger = MetricLogger(tb_writer)

histogram_interval = get_histogram_interval()

print_run_summary(run_id, config, start_time)

validate_config(config)

data_path = config.get('data_path', args.data)
train_data, val_data, text = load_data(data_path)
print(f"Loaded {len(text)} characters of training data")

encode, decode, vocab_size = create_tokenizer(text)
print(f"Vocabulary size: {vocab_size}")

model_config_params = {k: v for k, v in config.items() 
                       if k not in ['data_path', 'train_split', 'val_split', 'aim_logging', 'torch_compile']}
config_obj = ModelConfig.from_dict(model_config_params, vocab_size)
config_obj.validate()
config_obj.apply_to_model_module(model)

batch_size = config_obj.batch_size
block_size = config_obj.block_size
max_iters = config_obj.max_iters
eval_interval = config_obj.eval_interval
learning_rate = config_obj.learning_rate
eval_iters = config_obj.eval_iters
n_embd = config_obj.n_embd
n_head = config_obj.n_head
n_layer = config_obj.n_layer
dropout = config_obj.dropout

device = config_obj.device
print(f"Using device: {device}")

model_instance = LanguageModel()
m = model_instance.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

use_compile = config.get('torch_compile', True)
if use_compile and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile for optimized performance...")
    m = torch.compile(m, backend='aot_eager')
    print("✓ Model compiled successfully")
else:
    if not use_compile:
        print("torch.compile disabled (torch_compile=false in config)")
    else:
        print("torch.compile not available (requires PyTorch 2.0+)")

@torch.no_grad()
def estimate_loss() -> dict:
    """
    Estimate average loss on training and validation sets.
    
    This function evaluates the model on multiple batches to get a stable
    estimate of the loss, which is less noisy than single-batch evaluation.
    The @torch.no_grad() decorator disables gradient computation for efficiency.
    """
    out = {}
    model_instance.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            logits, loss = model_instance.prefill(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model_instance.train()
    return out

optimizer = torch.optim.AdamW(model_instance.parameters(), lr=learning_rate)

try:
    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if aim_tracker is not None:
                aim_tracker.track_metric('loss', losses['train'].item(), iter, context={'subset': 'train'})
                aim_tracker.track_metric('loss', losses['val'].item(), iter, context={'subset': 'val'})
            
            tb_logger.log_evaluation_metrics(
                losses['train'].item(),
                losses['val'].item(),
                learning_rate,
                iter
            )
        
        if histogram_interval > 0 and (iter % histogram_interval == 0 or iter == max_iters - 1):
            tb_logger.log_model_histograms(model_instance, iter)
            
            if aim_tracker is not None:
                for name, param in model_instance.named_parameters():
                    aim_tracker.track_distribution(name, param.detach().cpu().numpy(), iter, context={'type': 'param'})
                    
                    if param.grad is not None:
                        aim_tracker.track_distribution(name + '.grad', param.grad.detach().cpu().numpy(), iter, context={'type': 'grad'})

        xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device)

        logits, loss = model_instance.prefill(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if aim_tracker is not None:
            aim_tracker.track_metric('loss', loss.item(), iter, context={'subset': 'train'})
        
        tb_logger.log_training_loss(loss.item(), iter)

finally:
    if aim_tracker is not None:
        aim_tracker.close()

tb_writer.close()

print_tensorboard_instructions(tensorboard_dir)

print("\nTraining complete! Saving model weights...")
end_time = datetime.now().isoformat()

model_path = os.path.join(output_paths['export'], 'model.safetensors')
config_path = os.path.join(output_paths['export'], 'config.json')

model_instance.save_safetensors(model_path)
print(f"Model weights saved to '{model_path}' in SafeTensors format")

import json
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuration saved to '{config_path}'")

if aim_enabled:
    print_aim_instructions()

print_completion_summary(run_id, config, start_time, end_time)
