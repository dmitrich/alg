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

# Added for modular imports - PyTorch core modules
import torch

# Added for CLI support - command-line argument parsing
import argparse

# Added for modular imports - configuration management
from config import load_config, validate_config

# Added for modular imports - parameter management
from parameters import ModelConfig

# Added for modular imports - tokenization utilities
from utils import create_tokenizer

# Added for modular imports - model architecture
import model
from model import BigramLanguageModel

# ============================================================================
# STEP 1: Parse Command-Line Arguments
# ============================================================================
# Added for CLI support - parse command-line arguments
# This allows users to customize inference behavior without modifying code
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
args = parser.parse_args()

# ============================================================================
# STEP 2: Device Selection
# ============================================================================
# Added for device selection - automatically use GPU if available, fallback to CPU
# This ensures the script works on both GPU and CPU-only systems
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# STEP 3: Load Configuration
# ============================================================================
# Added for configuration loading - load hyperparameters from config.json
# These hyperparameters must match the ones used during training
config = load_config(args.config)
validate_config(config)

# Display configuration parameters in a formatted table
print("\n" + "=" * 50)
print("Configuration Parameters")
print("=" * 50)
for param_name, param_value in config.items():
    print(f"{param_name:<20} {param_value}")
print("=" * 50 + "\n")

# ============================================================================
# STEP 4: Build Vocabulary from Training Data
# ============================================================================
# Added for tokenizer creation - load training data to build vocabulary
# IMPORTANT: We must use the SAME vocabulary as training for correct token mapping
# The vocabulary is derived from all unique characters in the training data
with open(args.data, 'r', encoding='utf-8') as f:
    text = f.read()

# Added for tokenizer creation - create character-level tokenizer
# This creates encode/decode functions and determines vocabulary size
# encode: converts text string to list of integer token IDs
# decode: converts list of integer token IDs back to text string
encode, decode, vocab_size = create_tokenizer(text)
print(f"Vocabulary size: {vocab_size}")

# ============================================================================
# STEP 5: Create ModelConfig Instance
# ============================================================================
# Added for parameter management - create ModelConfig instance from config dict
# This provides type-safe parameter management and validation
config_obj = ModelConfig.from_dict(config, vocab_size)
config_obj.validate()  # Validate all parameters
config_obj.apply_to_model_module(model)  # Set global variables in model module

# ============================================================================
# STEP 6: Initialize Model Architecture
# ============================================================================
# Added for model initialization - instantiate the BigramLanguageModel
# This creates the model with the architecture defined by the hyperparameters
m = BigramLanguageModel()
m = m.to(config_obj.device)  # Move model to GPU or CPU
print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

# ============================================================================
# STEP 7: Load Trained Weights
# ============================================================================
# Added for model loading - load trained weights from SafeTensors file
# This restores the model parameters that were learned during training
# SafeTensors format provides fast, safe, and deterministic loading
print("Loading model weights...")
m.load_safetensors(args.model_path)
print(f"Model weights loaded successfully from {args.model_path}!")

# Added for inference mode - set model to evaluation mode
# This disables dropout and other training-specific behaviors
m.eval()

# ============================================================================
# STEP 8: Encode User Prompt
# ============================================================================
# Added for CLI support - encode the user-provided prompt
# Convert the text prompt into a sequence of token IDs that the model can process
context_text = args.prompt
context_tokens = encode(context_text)  # List of integer token IDs
context = torch.tensor([context_tokens], dtype=torch.long, device=config_obj.device)  # Shape: (1, prompt_length)

# ============================================================================
# STEP 9: Generate Text
# ============================================================================
# Generate text - using the same generation logic as the notebook
# The model.generate() method:
# 1. Takes the context tokens as input
# 2. Repeatedly predicts the next token
# 3. Appends predicted token to context
# 4. Continues until max_new_tokens are generated
# 5. Returns the full sequence (prompt + generated tokens)
print("\nGenerating text...")
print("=" * 80)
print(f"Prompt: {repr(context_text)}")
print("-" * 80)

# Call the generate method with torch.no_grad() for efficiency (no gradient computation needed)
with torch.no_grad():
    generated_tokens = m.generate(context, max_new_tokens=args.max_tokens)[0].tolist()

# ============================================================================
# STEP 10: Decode and Display Generated Text
# ============================================================================
# Convert the generated token IDs back to human-readable text
generated_text = decode(generated_tokens)
print(generated_text)
print("=" * 80)
