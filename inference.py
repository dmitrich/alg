"""
Inference script for GPT model.
Loads trained model weights and generates text from prompts.

Usage:
    python inference.py --prompt "Hello" --max_tokens 500
    python inference.py --prompt "ROMEO:" --max_tokens 1000 --model_path model.safetensors
    
The script performs the following steps:
1. Load configuration from config.json
2. Load training data to build vocabulary (same as training)
3. Create character-level tokenizer
4. Create ModelConfig instance for type-safe parameter management
5. Initialize model architecture with config hyperparameters
6. Load trained weights from SafeTensors file
7. Encode user prompt to token IDs
8. Generate new tokens using the model
9. Decode generated tokens back to text
"""

import torch

import argparse

from config import load_config, validate_config

from parameters import ModelConfig

from tokenizer import create_tokenizer

import model
from model import LanguageModel

from utils.alg.utils_config_loader import ConfigLoader
from utils.alg.utils_output import print_inference_summary, print_inference_completion
from datetime import datetime
import os
import json
import sys

parser = argparse.ArgumentParser(description='Generate text using a trained GPT model')
parser.add_argument('--prompt', type=str, default='\n',
                    help='Starting prompt for text generation (default: newline character)')
parser.add_argument('--max_tokens', type=int, default=500,
                    help='Maximum number of tokens to generate (default: 500)')
parser.add_argument('--model_path', type=str, default='model.safetensors',
                    help='Path to SafeTensors model file (default: model.safetensors)')
parser.add_argument('--config', type=str, default='config.json',
                    help='Path to configuration file (default: config.json)')
parser.add_argument('--data', type=str, default='data/input.txt',
                    help='Path to training data for vocabulary (default: data/input.txt)')
parser.add_argument('--run-id', type=str, default=None,
                    help='Run ID to load model from (e.g., "2024-01-15_001_baseline")')
args = parser.parse_args()

def get_latest_run_id(runs_dir: str = "runs") -> str:
    """
    Get the most recent run ID from runs directory.
    
    Args:
        runs_dir: Directory containing run directories
        
    Returns:
        Latest run ID string (e.g., "2024-01-15_003_baseline")
    """
    if not os.path.exists(runs_dir):
        return None
    
    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not run_dirs:
        return None
    
    run_dirs.sort(reverse=True)
    return run_dirs[0]

def resolve_model_paths(args) -> tuple:
    """
    Resolve model and config paths based on arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (model_path, config_path, model_id)
        
    Priority:
        1. Direct paths (--model_path, --config)
        2. Run directory (--run-id)
        3. Latest run (if runs/ exists)
        4. Root-level files (backward compatibility)
    """
    if args.model_path != 'model.safetensors' or args.config != 'config.json':
        return args.model_path, args.config, args.model_path
    
    if args.run_id:
        run_dir = os.path.join('runs', args.run_id)
        model_path = os.path.join(run_dir, 'artifacts', 'export', 'model.safetensors')
        config_path = os.path.join(run_dir, 'artifacts', 'export', 'config.json')
        return model_path, config_path, args.run_id
    
    latest_run_id = get_latest_run_id()
    if latest_run_id:
        run_dir = os.path.join('runs', latest_run_id)
        model_path = os.path.join(run_dir, 'artifacts', 'export', 'model.safetensors')
        config_path = os.path.join(run_dir, 'artifacts', 'export', 'config.json')
        return model_path, config_path, latest_run_id
    
    return 'model.safetensors', 'config.json', 'root'

def get_prompt_from_user() -> str:
    """
    Prompt user for input text.
    
    Returns:
        User's prompt string, or None if user wants to abort
        
    Behavior:
        - Prints "please provide a prompt: "
        - Reads user input from console
        - Returns None if input is "no", "stop", or "quit"
        - Returns input string otherwise
    """
    user_input = input("please provide a prompt: ").strip()
    if user_input.lower() in ['no', 'stop', 'quit']:
        return None
    return user_input if user_input else '\n'

def ask_for_another_prompt() -> bool:
    """
    Ask user if they want to provide another prompt.
    
    Returns:
        True if user wants to continue, False if user wants to exit
        
    Behavior:
        - Prints "another prompt? "
        - Reads user input from console
        - Returns False if input is "no", "stop", or "quit"
        - Returns True otherwise
    """
    user_input = input("\nanother prompt? ").strip()
    return user_input.lower() not in ['no', 'stop', 'quit']

def run_interactive_inference(m, encode, decode, config_obj, args):
    """
    Run inference in interactive loop.
    
    Args:
        m: Loaded model
        encode: Tokenizer encode function
        decode: Tokenizer decode function
        config_obj: ModelConfig instance
        args: Command-line arguments
        
    Behavior:
        1. Get prompt from user (abort if None)
        2. Run inference with prompt
        3. Ask if user wants another prompt
        4. If yes, repeat from step 1
        5. If no, exit
    """
    while True:
        prompt = get_prompt_from_user()
        if prompt is None:
            print("Inference aborted.")
            return
        
        print("\nGenerating text...")
        print("=" * 80)
        print(f"Prompt: {repr(prompt)}")
        print("-" * 80)
        
        context_tokens = encode(prompt)
        context = torch.tensor([context_tokens], dtype=torch.long, device=config_obj.device)
        
        with torch.no_grad():
            generated_tokens = m.decode(context, max_new_tokens=args.max_tokens)[0].tolist()
        
        generated_text = decode(generated_tokens)
        print(generated_text)
        print("=" * 80)
        
        if not ask_for_another_prompt():
            print("Exiting inference.")
            return

model_path, config_path, model_id = resolve_model_paths(args)
print(f"Loading model from: {model_id}")

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

start_time = datetime.now().isoformat()

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = load_config('config.json')

validate_config(config)

print_inference_summary(model_id, config, start_time)

data_path = config.get('data_path', args.data)
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

encode, decode, vocab_size = create_tokenizer(text)
print(f"Vocabulary size: {vocab_size}")

model_config_params = {k: v for k, v in config.items() 
                       if k not in ['data_path', 'train_split', 'val_split', 'aim_logging', 'torch_compile']}
config_obj = ModelConfig.from_dict(model_config_params, vocab_size)
config_obj.validate()
config_obj.apply_to_model_module(model)

m = LanguageModel()
m = m.to(config_obj.device)
print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

print("Loading model weights...")
m.load_safetensors(model_path)
print(f"Model weights loaded successfully from {model_path}!")

m.eval()

use_compile = config.get('torch_compile', True)
if use_compile and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile for optimized inference...")
    m = torch.compile(m, backend='aot_eager')
    print("✓ Model compiled successfully")
else:
    if not use_compile:
        print("torch.compile disabled (torch_compile=false in config)")
    else:
        print("torch.compile not available (requires PyTorch 2.0+)")

if args.prompt == '\n' and not any(arg.startswith('--prompt') for arg in sys.argv):
    run_interactive_inference(m, encode, decode, config_obj, args)
else:
    context_text = args.prompt
    context_tokens = encode(context_text)
    context = torch.tensor([context_tokens], dtype=torch.long, device=config_obj.device)
    
    print("\nGenerating text...")
    print("=" * 80)
    print(f"Prompt: {repr(context_text)}")
    print("-" * 80)
    
    with torch.no_grad():
        generated_tokens = m.decode(context, max_new_tokens=args.max_tokens)[0].tolist()
    
    generated_text = decode(generated_tokens)
    print(generated_text)
    print("=" * 80)

end_time = datetime.now().isoformat()
print_inference_completion(model_id, config, start_time, end_time)
