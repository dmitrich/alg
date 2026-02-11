# ALG1 Architecture Documentation

This document provides a comprehensive overview of the ALG1 project architecture, component interactions, and design decisions.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Training Pipeline](#training-pipeline)
- [Configuration Management](#configuration-management)
- [Experiment Tracking](#experiment-tracking)
- [Logging and Visualization](#logging-and-visualization)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Extensibility](#extensibility)
- [Deployment](#deployment)

---

## Overview

ALG1 is a character-level GPT (Generative Pre-trained Transformer) training framework with a focus on:

- **Modularity**: Each component is self-contained and can be developed/tested independently
- **Experiment Tracking**: Comprehensive tracking of hyperparameters, metrics, and metadata
- **Reproducibility**: Automatic capture of git state, environment, and configuration
- **Visualization**: Real-time monitoring with TensorBoard and post-hoc analysis with Aim
- **Production-Ready**: Safe model serialization with SafeTensors format

The project implements a decoder-only transformer architecture for character-level language modeling, trained on text data (typically Shakespeare dataset by default).

---

## Project Structure

```
alg1/
├── train.py                    # Main training script
├── model.py                    # GPT model implementation
├── data.py                     # Data loading and batch generation
├── config.py                   # Legacy configuration loader
├── parameters.py               # Type-safe configuration management
├── utils.py                    # Tokenization utilities
├── inference.py                # Inference script
├── pyproject.toml              # Project dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview
├── README-AIM.md               # Aim experiment tracking guide
├── README-TENSORBOARD.md       # TensorBoard visualization guide
├── README-ARCH-QWEN3.md        # This architecture document
├── configs/                    # Configuration files
│   ├── model.json              # Model architecture parameters
│   ├── train.yaml              # Training hyperparameters
│   ├── data.yaml               # Data configuration
│   └── tensorboard.json        # TensorBoard settings
├── data/                       # Training data (gitignored)
│   └── input.txt               # Shakespeare dataset
├── utils_alg1/                 # ALG1 utilities
│   ├── __init__.py
│   ├── utils_config_loader.py  # Multi-source config loading
│   ├── utils_run_manager.py    # Run directory and metadata management
│   └── utils_output.py         # Console output formatting
├── utils_tensorboard/          # TensorBoard integration
│   ├── __init__.py
│   ├── config.py               # TensorBoard configuration
│   ├── writer.py               # Writer lifecycle management
│   ├── logger.py               # Metric logging utilities
│   ├── instructions.py         # UI launch instructions
│   └── server.py               # TensorBoard server management
├── utils_aim/                  # Aim experiment tracking
│   ├── __init__.py
│   ├── tracker.py              # AimTracker wrapper class
│   └── instructions.py         # UI launch instructions
├── runs/                       # Training runs (gitignored)
│   └── YYYY-MM-DD_NNN_tag/     # Timestamped run directories
│       ├── meta/               # Run metadata
│       │   ├── full_config.json
│       │   ├── git.txt
│       │   ├── env.txt
│       │   └── notes.md
│       ├── logs/
│       │   └── tensorboard/    # TensorBoard event files
│       ├── artifacts/
│       │   ├── checkpoints/    # Model checkpoints
│       │   └── export/         # Final model export
│       └── eval/               # Evaluation results
└── .aim/                       # Aim experiment repository (gitignored)
```

---

## Core Components

### 1. Model (`model.py`)

**BigramLanguageModel** - A character-level transformer language model.

**Architecture:**
```
Input (B, T) → Token Embedding (B, T, n_embd)
             → Position Embedding (T, n_embd)
             → Transformer Blocks (n_layer × [SA + FFN])
             → Layer Norm
             → LM Head → Logits (B, T, vocab_size)
```

**Key Components:**
- **Head**: Single attention head with Q, K, V projections
- **MultiHeadAttention**: Parallel attention heads with output projection
- **FeedFoward**: MLP with ReLU activation
- **Block**: Transformer block with self-attention and feed-forward layers
- **BigramLanguageModel**: Main model class with token/position embeddings

**Features:**
- Multi-head self-attention
- Feed-forward networks with 4× expansion
- Layer normalization
- Dropout regularization
- SafeTensors serialization

**Methods:**
- `forward(idx, targets)`: Compute logits and loss
- `generate(idx, max_new_tokens)`: Autoregressive text generation
- `save_safetensors(path)`: Save weights in SafeTensors format
- `load_safetensors(path)`: Load weights from SafeTensors format

### 2. Data (`data.py`)

**Functions:**
- `load_data(data_path)`: Load and split training data (90/10 train/val)
- `get_batch(split, train_data, val_data, block_size, batch_size, device)`: Generate training batches

**Features:**
- Automatic data splitting
- Device-aware tensor placement
- Random sequence sampling
- Shifted target generation (next-token prediction)

### 3. Tokenization (`utils.py`)

**Function:**
- `create_tokenizer(text)`: Create character-level tokenizer

**Returns:**
- `encode(s)`: String → list of integers
- `decode(l)`: List of integers → string
- `vocab_size`: Number of unique characters

**Features:**
- Character-level tokenization
- Sorted vocabulary for determinism
- Bidirectional mappings (stoi, itos)

### 4. Configuration Management

**Legacy (`config.py`):**
- `load_config(path)`: Load from single JSON file
- `validate_config(config)`: Validate all parameters

**Modern (`utils_alg1/utils_config_loader.py`):**
- `ConfigLoader`: Load and merge multiple config sources
  - `load_configs(config_dir)`: Load all configs from directory
  - `load_model_config(path)`: Load model architecture
  - `load_train_config(path)`: Load training hyperparameters
  - `load_data_config(path)`: Load data configuration

**Type-Safe (`parameters.py`):**
- `ModelConfig` dataclass with validation
- `from_json(path, vocab_size)`: Load from JSON file
- `from_dict(config_dict, vocab_size)`: Create from dictionary
- `validate()`: Comprehensive parameter validation
- `apply_to_model_module(model)`: Set global variables for backward compatibility

**Configuration Sources:**
1. `configs/model.json` - Model architecture (n_embd, n_head, n_layer, dropout, vocab_size)
2. `configs/train.yaml` - Training parameters (batch_size, block_size, max_iters, etc.)
3. `configs/data.yaml` - Data configuration (data_path, train_split, val_split)
4. `configs/tensorboard.json` - TensorBoard settings (histogram_interval, port)

### 5. Run Management (`utils_alg1/utils_run_manager.py`)

**RunIDGenerator:**
- `generate(tag)`: Generate unique run ID (YYYY-MM-DD_NNN_tag format)
- `get_next_counter(date)`: Get next counter for date

**RunDirectoryCreator:**
- `create(run_id)`: Create run directory structure
- `create_subdirectories(run_dir)`: Create meta, logs, artifacts, eval subdirs

**MetadataCapture:**
- `get_git_commit()`: Get current git commit hash
- `get_git_branch()`: Get current git branch
- `get_python_version()`: Get Python version
- `capture_git_info(path)`: Write git info to file
- `capture_env_info(path)`: Write environment info to file
- `copy_configs(config_dir, meta_dir)`: Copy configs to run directory

---

## Training Pipeline

### Step-by-Step Flow

```
1. Parse CLI arguments
   ↓
2. Load and merge configurations
   ↓
3. Generate unique run ID
   ↓
4. Create run directory structure
   ↓
5. Capture metadata (git, environment)
   ↓
6. Initialize Aim tracker
   ↓
7. Initialize TensorBoard writer
   ↓
8. Load and tokenize training data
   ↓
9. Create ModelConfig instance
   ↓
10. Initialize model
   ↓
11. Training loop:
    - Sample batch
    - Forward pass
    - Backward pass
    - Update weights
    - Track metrics (Aim + TensorBoard)
    - Periodic evaluation
    - Periodic histogram logging
   ↓
12. Save model weights (SafeTensors)
   ↓
13. Cleanup (close writers, finalize Aim)
   ↓
14. Print completion summary
```

### Training Loop Details

**Every iteration:**
1. Sample batch from training data
2. Forward pass: compute logits and loss
3. Zero gradients
4. Backward pass: compute gradients
5. Update weights with optimizer
6. Track training loss (Aim + TensorBoard)

**Every eval_interval iterations:**
1. Evaluate loss on train and val sets
2. Track evaluation metrics (Aim + TensorBoard)
3. Log evaluation metrics to TensorBoard

**Every histogram_interval iterations:**
1. Log parameter histograms to TensorBoard
2. Log gradient histograms to TensorBoard
3. Track parameter distributions (Aim)
4. Track gradient distributions (Aim)

### CLI Arguments

```
--config: Path to config file (default: config.json)
--data: Path to training data (default: data/input.txt)
--output: Output model path (default: model.safetensors)
--batch-size: Override batch_size
--block-size: Override block_size
--max-iters: Override max_iters
--learning-rate: Override learning_rate
--eval-interval: Override eval_interval
--eval-iters: Override eval_iters
--n-embd: Override n_embd
--n-head: Override n_head
--n-layer: Override n_layer
--dropout: Override dropout
--run-tag: Append tag to run ID
```

---

## Configuration Management

### Configuration Loading Strategy

**Priority Order:**
1. CLI arguments (highest priority)
2. `configs/` directory files
3. Legacy `config.json` (fallback)

**Merging Process:**
```python
merged_config = {}
merged_config.update(model_config)      # From model.json
merged_config.update(train_config)      # From train.yaml
merged_config.update(data_config)       # From data.yaml
# Override with CLI arguments
```

### Configuration Validation

**ModelConfig validation checks:**
- All numeric parameters > 0
- Dropout in range [0, 1)
- n_embd divisible by n_head
- All required parameters present

**Error handling:**
- Clear error messages for missing parameters
- Descriptive validation failures
- Early failure before training begins

---

## Experiment Tracking

### Aim Integration

**Purpose:** Comprehensive experiment tracking with UI for comparison and analysis.

**Features:**
- Hyperparameter tracking
- Metric tracking with context (train/val)
- Distribution tracking (parameters and gradients)
- Git and environment metadata
- Run directory linking

**AimTracker Class:**
```python
aim_tracker = AimTracker(run_name, run_dir)
aim_tracker.track_config(config)           # Track hyperparameters
aim_tracker.track_metadata('git_commit', commit)  # Track metadata
aim_tracker.track_metric('loss', value, step, context)  # Track metrics
aim_tracker.track_distribution(name, values, step, context)  # Track distributions
aim_tracker.close()                         # Cleanup
```

**Error Handling:**
- All operations wrapped in try-except
- Training continues if Aim fails
- Warning messages for failures

### TensorBoard Integration

**Purpose:** Real-time training monitoring and visualization.

**Features:**
- Training loss tracking
- Evaluation metrics (train/val loss)
- Learning rate monitoring
- Parameter and gradient histograms
- Graceful degradation if TensorBoard unavailable

**MetricLogger Class:**
```python
tb_logger.log_training_loss(loss, step)
tb_logger.log_evaluation_metrics(train_loss, val_loss, lr, step)
tb_logger.log_model_histograms(model, step)
```

### Metadata Capture

**Git Information:**
- Commit hash
- Branch name
- Dirty flag (uncommitted changes)

**Environment Information:**
- Python version
- PyTorch version

**Configuration:**
- Full merged config saved to `meta/full_config.json`
- Original config files copied to `meta/`

---

## Logging and Visualization

### TensorBoard

**Event Files:**
- `events.out.tfevents.*` - Training events

**Logged Metrics:**
- `Loss/train_step` - Training loss per step
- `Loss/train` - Average training loss (eval)
- `Loss/val` - Average validation loss (eval)
- `Learning_Rate` - Current learning rate
- `Parameters/*` - Parameter histograms
- `Gradients/*` - Gradient histograms

**Launch:**
```bash
tensorboard --logdir=runs --port=6006
```

### Aim

**Repository Structure:**
- `.aim/` - Repository directory
- Run data stored in subdirectories

**Tracked Data:**
- Metrics (loss with context)
- Hyperparameters (all config values)
- Distributions (parameters and gradients)
- Metadata (git, environment, run_dir)

**Launch:**
```bash
aim up
```

---

## Data Flow

### Training Data Flow

```
data/input.txt
    ↓ (load_data)
torch.Tensor (encoded integers)
    ↓ (split 90/10)
train_data, val_data
    ↓ (get_batch)
xb (B, T), yb (B, T)
    ↓ (model.forward)
logits (B, T, vocab_size), loss
    ↓ (backward)
gradients
    ↓ (optimizer.step)
updated weights
```

### Configuration Flow

```
CLI args + configs/
    ↓ (merge)
merged_config
    ↓ (ModelConfig.from_dict)
config_obj (validated)
    ↓ (apply_to_model_module)
model module globals
    ↓
model initialization
```

### Metadata Flow

```
Run start
    ↓ (MetadataCapture)
git commit, branch, Python version
    ↓ (save to meta/)
git.txt, env.txt
    ↓ (AimTracker)
track_metadata()
    ↓
Aim Run object
```

---

## Design Patterns

### 1. Module Separation

Each component is in its own module:
- `model.py` - Model architecture
- `data.py` - Data loading
- `config.py` - Configuration
- `utils.py` - Utilities

**Benefits:**
- Independent testing
- Reusability
- Clear responsibilities

### 2. Wrapper Pattern

Aim and TensorBoard use wrapper classes:
- `AimTracker` - Encapsulates Aim SDK
- `TensorBoardWriter` - Manages SummaryWriter lifecycle
- `MetricLogger` - High-level logging methods

**Benefits:**
- Fail-safe error handling
- Consistent interface
- Easy to replace underlying implementation

### 3. Dataclass for Configuration

`ModelConfig` uses Python dataclass:
- Type hints for IDE support
- Automatic validation
- Easy serialization

**Benefits:**
- Type safety
- IDE autocomplete
- Clear structure

### 4. Factory Pattern

Configuration loading uses factory methods:
- `ModelConfig.from_json()`
- `ModelConfig.from_dict()`

**Benefits:**
- Flexible creation
- Validation built-in
- Clear intent

### 5. Strategy Pattern

Multiple configuration sources:
- Legacy JSON
- New YAML/JSON split
- CLI overrides

**Benefits:**
- Backward compatibility
- Flexibility
- Easy migration

---

## Extensibility

### Adding New Metrics

1. Add tracking call in training loop:
```python
aim_tracker.track_metric('metric_name', value, step, context)
tb_logger.log_training_loss(value, step)
```

### Adding New Configuration Parameters

1. Add to `configs/model.json` or `configs/train.yaml`
2. Update `ModelConfig` dataclass
3. Update validation in `ModelConfig.validate()`

### Adding New Logging Destinations

1. Create new tracker class (similar to `AimTracker`)
2. Initialize in `train.py`
3. Add tracking calls in training loop

### Adding New Data Sources

1. Modify `data.py` or create new data module
2. Implement `load_data()` and `get_batch()` functions
3. Update training loop to use new data

---

## Deployment

### Local Development

1. Install dependencies:
```bash
uv sync
```

2. Run training:
```bash
python train.py
```

3. View TensorBoard:
```bash
tensorboard --logdir=runs --port=6006
```

4. View Aim:
```bash
aim up
```

### Production Deployment

**Model Export:**
- Model saved to `runs/.../export/model.safetensors`
- Config saved to `runs/.../export/config.json`

**Inference:**
```python
from model import BigramLanguageModel
model = BigramLanguageModel()
model.load_safetensors('model.safetensors')
model.eval()
```

### CI/CD Considerations

**Testing:**
- Unit tests in `tests/` directory
- Property-based tests with Hypothesis
- Integration tests for full pipeline

**Environment:**
- Python 3.12+
- PyTorch 2.2.2+
- GPU recommended for training

---

## Troubleshooting

### Common Issues

**1. Aim not found:**
```bash
pip install aim
```

**2. TensorBoard not found:**
```bash
pip install tensorboard
```

**3. Data file not found:**
- Ensure `data/input.txt` exists
- Or specify `--data` argument

**4. Git info unavailable:**
- Initialize git repo: `git init`
- Or use placeholder values (automatically handled)

### Debug Mode

**Enable verbose logging:**
```python
# Add to train.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check configuration:**
```python
print(config)  # Print merged config
```

**Verify Aim tracking:**
```python
print(aim_tracker.enabled)  # Should be True
```

---

## Performance Considerations

### Training Speed

**GPU vs CPU:**
- GPU: ~100× faster than CPU
- Auto-detects CUDA availability
- Falls back to CPU if no GPU

**Batch Size:**
- Larger batches = faster training (up to memory limit)
- Default: 4 (for CPU compatibility)
- GPU: Can use larger batches (8, 16, 32+)

**Evaluation Interval:**
- Lower = more frequent evaluation = slower training
- Default: 100 iterations
- Adjust based on dataset size

### Memory Optimization

**Gradient Accumulation:**
- Not currently implemented
- Can be added for memory-constrained environments

**Mixed Precision:**
- Not currently implemented
- Can be added with `torch.cuda.amp`

**Model Size:**
- Default: 64 embedding dim, 4 layers
- Can be scaled up/down by changing config

---

## Future Enhancements

### Planned Features

1. **Learning Rate Schedulers**
   - Cosine decay
   - Warmup
   - Step decay

2. **Gradient Clipping**
   - Prevent exploding gradients
   - Configurable threshold

3. **Checkpointing**
   - Save checkpoints at intervals
   - Resume from checkpoint

4. **Distributed Training**
   - Multi-GPU support
   - Data parallelism

5. **Advanced Logging**
   - Activation histograms
   - Attention weights
   - Feature visualizations

6. **Hyperparameter Search**
   - Grid search
   - Random search
   - Bayesian optimization

---

## Contributing

### Code Style

- Follow existing module structure
- Add docstrings to new functions
- Use type hints
- Handle errors gracefully

### Testing

- Add unit tests for new features
- Add property-based tests for core logic
- Test error scenarios

### Documentation

- Update this document for architecture changes
- Update README files for user-facing changes
- Add inline comments for complex logic

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- GPT architecture based on "Attention Is All You Need" (Vaswani et al.)
- SafeTensors for secure model serialization
- Aim and TensorBoard for experiment tracking and visualization

---

**Last Updated:** February 10, 2026
**Author:** Qwen3 Coder Next
