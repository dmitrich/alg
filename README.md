# GPT Character-Level Language Model

A modular implementation of a character-level GPT (Generative Pre-trained Transformer) model with modern SafeTensors format for model persistence. This project demonstrates transformer architecture with multi-head self-attention, training on Shakespeare text, and text generation capabilities.

## Features

- **Modern Architecture**: Multi-head self-attention, transformer blocks, layer normalization
- **SafeTensors Format**: Fast, secure, and industry-standard model serialization
- **Modular Design**: Separate training and inference modules with clean interfaces
- **Character-Level Tokenization**: Simple and effective text processing
- **Configurable**: Easy hyperparameter tuning via JSON configuration
- **Migration Support**: Tools to migrate from legacy text-based weight files

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.1+
- safetensors 0.4.0+

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install torch safetensors

# Or using uv (recommended)
uv pip install torch safetensors
```

## Quick Start

### Training a Model

Train a model on your text data:

```bash
python train.py --data data/input.txt --config config.json --output model.safetensors
```

The training script will:
1. Load training data from `data/input.txt`
2. Initialize model with parameters from `config.json`
3. Train the model with progress updates
4. Save trained weights to `model.safetensors`

### Generating Text

Generate text using a trained model:

```bash
python inference.py --model_path model.safetensors --prompt "First Citizen:" --max_tokens 500
```

Example output:
```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
...
```

## Configuration

Model hyperparameters are defined in `config.json`:

```json
{
  "batch_size": 4,
  "block_size": 256,
  "max_iters": 3000,
  "eval_interval": 100,
  "learning_rate": 0.01,
  "eval_iters": 200,
  "n_embd": 128,
  "n_head": 4,
  "n_layer": 4,
  "dropout": 0.0
}
```

### Parameters

- **batch_size**: Number of sequences processed in parallel
- **block_size**: Maximum context length (sequence length)
- **max_iters**: Total training iterations
- **eval_interval**: Evaluate loss every N iterations
- **learning_rate**: AdamW optimizer learning rate
- **eval_iters**: Number of batches for evaluation
- **n_embd**: Embedding dimension (must be divisible by n_head)
- **n_head**: Number of attention heads
- **n_layer**: Number of transformer blocks
- **dropout**: Dropout probability (0.0 to 0.99)

## SafeTensors Format

This project uses SafeTensors for model persistence, providing several advantages over traditional formats:

### Benefits

- **Security**: No arbitrary code execution during deserialization (unlike pickle)
- **Performance**: 10-100x faster loading via memory-mapped I/O
- **File Size**: 50-70% smaller than text-based formats
- **Compatibility**: Works with Hugging Face, PyTorch, and modern inference engines
- **Determinism**: Reproducible serialization across platforms

### Usage

#### Saving Models

```python
from model import BigramLanguageModel

# Create and train model
model = BigramLanguageModel()
# ... training code ...

# Save to SafeTensors format
model.save_safetensors('model.safetensors')
```

#### Loading Models

```python
from model import BigramLanguageModel

# Create model instance
model = BigramLanguageModel()

# Load from SafeTensors format
model.load_safetensors('model.safetensors')
```

## Migration from Legacy Format

If you have existing text-based weight files, use the migration utility:

```bash
python migrate_weights.py --weights-dir weights --output model.safetensors --config config.json
```

The migration script will:
1. Load your old text-based weights
2. Convert to SafeTensors format
3. Verify the migration (all parameters match within 1e-6 tolerance)
4. Save the new format

See [MIGRATION.md](MIGRATION.md) for detailed migration instructions.

## Project Structure

```
.
├── model.py              # GPT model architecture
├── parameters.py         # Configuration management
├── train.py             # Training script
├── inference.py         # Text generation script
├── migrate_weights.py   # Migration utility
├── data.py              # Data loading utilities
├── utils.py             # Tokenization utilities
├── config.json          # Model hyperparameters
├── data/
│   └── input.txt        # Training data
└── model.safetensors    # Trained model weights
```

## Model Architecture

The model implements a standard transformer decoder architecture:

1. **Token & Position Embeddings**: Character-level token embeddings + learned positional embeddings
2. **Transformer Blocks**: Multi-head self-attention + feed-forward network + layer normalization
3. **Language Model Head**: Linear projection to vocabulary size with bias

### Architecture Details

- **Multi-Head Self-Attention**: Parallel attention heads with scaled dot-product attention
- **Feed-Forward Network**: Two-layer MLP with GELU activation
- **Layer Normalization**: Applied before attention and feed-forward layers
- **Dropout**: Applied to attention weights and feed-forward outputs

## Training Details

### Data Processing

- Character-level tokenization (no subword units)
- 90/10 train/validation split
- Random batch sampling with block_size context windows

### Optimization

- **Optimizer**: AdamW
- **Loss**: Cross-entropy loss on next-token prediction
- **Evaluation**: Periodic evaluation on validation set
- **Device**: Automatic GPU/CPU detection

### Training Progress

The training script displays:
- Current iteration and loss
- Estimated training time
- Validation loss at eval_interval
- Final model save confirmation

## Text Generation

The inference module supports:

- **Prompt-based generation**: Start with custom text
- **Configurable length**: Control max_tokens parameter
- **Temperature sampling**: Multinomial sampling from logits
- **Batch generation**: Generate multiple sequences (if needed)

## Command-Line Options

### Training

```bash
python train.py [OPTIONS]

Options:
  --data PATH          Path to training data (default: data/input.txt)
  --config PATH        Path to config file (default: config.json)
  --output PATH        Path to save model (default: model.safetensors)
  --max-iters INT      Override max_iters from config
  --learning-rate FLOAT Override learning_rate from config
```

### Inference

```bash
python inference.py [OPTIONS]

Options:
  --model_path PATH    Path to model file (default: model.safetensors)
  --config PATH        Path to config file (default: config.json)
  --data PATH          Path to training data for tokenizer (default: data/input.txt)
  --prompt TEXT        Starting text for generation (default: "\n")
  --max_tokens INT     Number of tokens to generate (default: 500)
```

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest

# Run specific test suites
pytest test_parameters.py      # Parameter validation tests
pytest test_model_safetensors.py  # SafeTensors save/load tests
pytest test_integration.py     # End-to-end integration tests
pytest test_properties.py      # Property-based tests

# Run with coverage
pytest --cov=. --cov-report=html
```

### Test Coverage

- **Unit Tests**: Parameter validation, model save/load, configuration management
- **Integration Tests**: Full training pipeline, inference pipeline, migration
- **Property-Based Tests**: Serialization determinism, lossless round-trip, inference equivalence

## Troubleshooting

### Common Issues

#### Out of Memory

If you encounter OOM errors during training:

```bash
# Reduce batch size
# Edit config.json: "batch_size": 2

# Or reduce model size
# Edit config.json: "n_embd": 64, "n_layer": 2
```

#### CUDA Not Available

The system automatically falls back to CPU if CUDA is unavailable. To force CPU:

```python
# In config.json
{
  "device": "cpu"
}
```

#### Model Loading Errors

If you get architecture mismatch errors:

1. Ensure `config.json` matches the model you're loading
2. Check that `vocab_size` matches your training data
3. Verify the model file is not corrupted

#### Slow Training

To speed up training:

1. Use GPU if available (automatic)
2. Increase batch_size (if memory allows)
3. Reduce eval_interval for fewer evaluations
4. Use smaller model (reduce n_embd, n_layer)

## Performance Tips

### Training Performance

- **GPU**: Training is 10-50x faster on GPU
- **Batch Size**: Larger batches improve GPU utilization
- **Mixed Precision**: Consider using torch.cuda.amp for faster training

### Inference Performance

- **Batch Generation**: Generate multiple sequences in parallel
- **Model Size**: Smaller models generate faster
- **SafeTensors**: Loading is near-instant with memory mapping

## Examples

### Training on Custom Data

```bash
# Prepare your text data
echo "Your custom text here..." > data/custom.txt

# Train model
python train.py --data data/custom.txt --output custom_model.safetensors

# Generate text
python inference.py --model_path custom_model.safetensors --data data/custom.txt --prompt "Start"
```

### Experimenting with Hyperparameters

```bash
# Create custom config
cp config.json config_large.json
# Edit config_large.json: increase n_embd, n_layer

# Train with custom config
python train.py --config config_large.json --output large_model.safetensors
```

### Batch Text Generation

```python
from model import BigramLanguageModel
import torch

# Load model
model = BigramLanguageModel()
model.load_safetensors('model.safetensors')
model.eval()

# Generate multiple sequences
prompts = torch.tensor([[0], [1], [2], [3]])  # Batch of 4 prompts
outputs = model.generate(prompts, max_new_tokens=100)

# Decode each sequence
for output in outputs:
    text = decode(output.tolist())
    print(text)
    print("---")
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Based on Andrej Karpathy's "Let's build GPT" tutorial
- Uses SafeTensors library by Hugging Face
- Trained on Shakespeare text from Project Gutenberg

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [SafeTensors Documentation](https://huggingface.co/docs/safetensors/index)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above
