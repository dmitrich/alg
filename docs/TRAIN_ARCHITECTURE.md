# train.py Architecture Documentation

## Overview

`train.py` is the main training script for the character-level GPT (Generative Pre-trained Transformer) model. It orchestrates the entire training pipeline from configuration loading to model persistence, with comprehensive experiment tracking and monitoring capabilities.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         train.py                                 │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │  Configuration │  │  Data Pipeline │  │  Model Training  │ │
│  │    Loading     │→ │   & Tokenizer  │→ │   & Evaluation   │ │
│  └────────────────┘  └────────────────┘  └──────────────────┘ │
│           ↓                                        ↓            │
│  ┌────────────────┐                    ┌──────────────────┐   │
│  │  Run Manager   │                    │  Experiment      │   │
│  │  & Metadata    │                    │  Tracking        │   │
│  └────────────────┘                    └──────────────────┘   │
│                                                  ↓              │
│                                     ┌──────────────────┐       │
│                                     │  Model           │       │
│                                     │  Persistence     │       │
│                                     └──────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Architectural Components

### 1. Configuration Management Layer

**Purpose**: Load, merge, and validate training configurations from multiple sources.

**Components**:
- `ConfigLoader`: Loads configurations from `configs/` directory (YAML/JSON files)
- `ModelConfig`: Type-safe dataclass for hyperparameters with validation
- CLI argument parser: Allows runtime overrides of configuration values

**Configuration Hierarchy** (highest to lowest priority):
1. CLI arguments (`--batch-size`, `--learning-rate`, etc.)
2. Config files in `configs/` directory (`train.yaml`, `model.json`)
3. Legacy `config.json` (backward compatibility)

**Key Features**:
- Automatic device detection (MPS → CUDA → CPU)
- Parameter validation before training starts
- Support for both new (`configs/`) and legacy (`config.json`) formats

### 2. Run Management System

**Purpose**: Create unique, reproducible training runs with comprehensive metadata capture.

**Components**:
- `RunIDGenerator`: Generates unique run identifiers (format: `YYYYMMDD_HHMMSS_<tag>`)
- `RunDirectoryCreator`: Creates structured output directories
- `MetadataCapture`: Captures git state, environment info, and configuration snapshots

**Directory Structure**:
```
runs/
└── YYYYMMDD_HHMMSS_<tag>/
    ├── artifacts/
    │   ├── checkpoints/     # Model checkpoints (future)
    │   └── export/          # Final trained model
    │       ├── model.safetensors
    │       └── config.json
    ├── logs/
    │   └── tensorboard/     # TensorBoard event files
    └── meta/
        ├── git.txt          # Git commit, branch, diff
        ├── env.txt          # Python version, packages
        └── full_config.json # Complete merged configuration
```

**Reproducibility Features**:
- Git commit hash and branch tracking
- Git diff capture (uncommitted changes)
- Python version and package versions
- Complete configuration snapshot

### 3. Data Pipeline

**Purpose**: Load, tokenize, and batch text data for training.

**Components**:
- `load_data()`: Loads text file and splits into train/val (90/10)
- `create_tokenizer()`: Creates character-level encoder/decoder
- `get_batch()`: Generates random batches from train or val split

**Data Flow**:
```
input.txt → load_data() → train_data, val_data
                ↓
         create_tokenizer() → encode(), decode(), vocab_size
                ↓
         get_batch() → (xb, yb) batches
                ↓
         model(xb, yb) → logits, loss
```

**Tokenization**:
- Character-level: Each unique character is a token
- Vocabulary: All unique characters in training data
- Encoding: `text → list[int]`
- Decoding: `list[int] → text`

### 4. Model Architecture Integration

**Purpose**: Initialize and configure the transformer model.

**Components**:
- `LanguageModel`: Decoder-only transformer with multi-head attention
- `ModelConfig`: Applies hyperparameters to model module (backward compatibility)
- `torch.compile()`: Optional PyTorch 2.x optimization

**Model Configuration**:
- `n_embd`: Embedding dimension (C)
- `n_head`: Number of attention heads
- `n_layer`: Number of transformer blocks
- `block_size`: Maximum context length (T)
- `dropout`: Regularization rate
- `vocab_size`: Number of unique tokens (V)

**Optimization**:
- `torch.compile()` with `aot_eager` backend for MPS compatibility
- Fuses operations and reduces Python overhead
- Significant speedup on Apple Silicon (M1/M2/M3/M4)

### 5. Training Loop

**Purpose**: Execute the core training algorithm with monitoring.

**Structure**:
```python
for iter in range(max_iters):
    # Evaluation (periodic)
    if iter % eval_interval == 0:
        losses = estimate_loss()  # Average over eval_iters batches
        log_to_tensorboard(losses)
        log_to_aim(losses)
    
    # Histogram logging (periodic)
    if iter % histogram_interval == 0:
        log_histograms(model_parameters, gradients)
    
    # Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Metric tracking
    log_training_loss(loss)
```

**Key Operations**:
1. **Forward pass**: `model(xb, yb)` → logits, loss
2. **Backward pass**: `loss.backward()` → compute gradients
3. **Optimizer step**: `optimizer.step()` → update parameters
4. **Evaluation**: Periodic loss estimation on train/val splits
5. **Logging**: Continuous metric tracking to TensorBoard and Aim

**Loss Estimation**:
- Averages loss over `eval_iters` batches (default: 200)
- Reduces noise from single-batch evaluation
- Computed separately for train and val splits
- Uses `@torch.no_grad()` for efficiency (no gradient computation)

#### Prefill Phase vs. Decode Phase

**Important Note**: The training loop in `train.py` operates in **Prefill mode only**. The Decode phase is used during inference (text generation), not training.

##### Prefill Phase (Training & Evaluation)

**What it is**: Processing an entire sequence in parallel to compute loss or generate initial context.

**In Training (`train.py`)**:
```python
# Input: Full sequence of tokens
xb = [[tok1, tok2, tok3, ..., tok64]]  # Shape: (B=4, T=64)
yb = [[tok2, tok3, tok4, ..., tok65]]  # Targets shifted by 1

# Forward pass: Process all T=64 tokens in parallel
logits, loss = model(xb, yb)  # Shape: (B=4, T=64, vocab_size=65)

# Compute loss for ALL positions simultaneously
# Position 0: predict tok2 from tok1
# Position 1: predict tok3 from tok1,tok2
# Position 2: predict tok4 from tok1,tok2,tok3
# ... all computed in one forward pass
```

**Characteristics**:
- **Parallelism**: All T positions processed simultaneously
- **Efficiency**: Leverages GPU parallelism (matrix operations)
- **Memory**: Requires O(T²) memory for attention (T×T attention matrix)
- **Speed**: Fast - single forward pass for entire sequence
- **Use case**: Training and evaluation (computing loss on known sequences)

**Computational Cost**:
- Attention: O(T² × C) for computing attention scores
- Feed-forward: O(T × C²) for each layer
- Total: Dominated by attention when T is large

**Example in Training**:
```
Input sequence:  "Hello world"
Tokens (xb):     [H, e, l, l, o, _, w, o, r, l, d]
Targets (yb):    [e, l, l, o, _, w, o, r, l, d, !]

Prefill processes all 11 positions in parallel:
- Position 0: [H] → predict 'e'
- Position 1: [H,e] → predict 'l'
- Position 2: [H,e,l] → predict 'l'
- ... (all computed simultaneously)
- Position 10: [H,e,l,l,o,_,w,o,r,l,d] → predict '!'

Loss = average cross-entropy across all 11 predictions
```

##### Decode Phase (Inference Only - NOT in train.py)

**What it is**: Generating tokens one at a time autoregressively.

**In Inference (`inference.py` or `model.generate()`)**:
```python
# Start with initial context
context = [tok1, tok2, tok3]  # Shape: (1, 3)

# Generate tokens one by one
for _ in range(max_new_tokens):
    # Step 1: Prefill - process current context
    logits, _ = model(context)  # Shape: (1, len(context), vocab_size)
    
    # Step 2: Sample next token from last position
    next_token = sample(logits[:, -1, :])  # Shape: (1, 1)
    
    # Step 3: Append to context for next iteration
    context = torch.cat([context, next_token], dim=1)
    
    # Repeat: context grows by 1 token each iteration
```

**Characteristics**:
- **Sequential**: One token generated per forward pass
- **Autoregressive**: Each token depends on all previous tokens
- **Memory**: O(T²) grows with sequence length
- **Speed**: Slow - requires N forward passes for N tokens
- **Use case**: Text generation, inference, chatbots

**Computational Cost**:
- First token: O(T² × C) - process initial context (prefill)
- Each subsequent token: O((T+k)² × C) where k is tokens generated
- Total for N tokens: O(N × T² × C) - much slower than prefill

**Example in Generation**:
```
Initial context: "Hello"
Tokens: [H, e, l, l, o]

Iteration 1 (Decode):
  Input: [H, e, l, l, o]
  Forward pass → logits for position 5
  Sample → ' ' (space)
  Context: [H, e, l, l, o, _]

Iteration 2 (Decode):
  Input: [H, e, l, l, o, _]
  Forward pass → logits for position 6
  Sample → 'w'
  Context: [H, e, l, l, o, _, w]

Iteration 3 (Decode):
  Input: [H, e, l, l, o, _, w]
  Forward pass → logits for position 7
  Sample → 'o'
  Context: [H, e, l, l, o, _, w, o]

... continues until max_new_tokens reached
```

##### Comparison Table

| Aspect | Prefill Phase | Decode Phase |
|--------|---------------|--------------|
| **Used in** | Training, Evaluation | Inference, Generation |
| **Processing** | Parallel (all tokens at once) | Sequential (one token at a time) |
| **Input shape** | (B, T) - full sequences | (B, T+k) - growing context |
| **Output** | Loss over all positions | Single next token |
| **Forward passes** | 1 per batch | N (for N generated tokens) |
| **Speed** | Fast (GPU parallelism) | Slow (sequential dependency) |
| **Memory** | O(T²) fixed | O((T+k)²) growing |
| **Attention mask** | Causal (lower triangular) | Causal (lower triangular) |
| **Optimization** | Batch processing | KV caching (not in this code) |
| **File** | `train.py` | `inference.py`, `model.generate()` |

##### Why Training Uses Prefill Only

**Teacher Forcing**: During training, we know the correct next token at every position, so we can:
1. Provide the entire input sequence at once
2. Compute predictions for all positions in parallel
3. Calculate loss across all positions simultaneously
4. Backpropagate through the entire sequence

This is much more efficient than generating tokens one by one.

**Example**:
```python
# Training (Prefill): 1 forward pass for 64 tokens
xb = torch.randint(0, vocab_size, (4, 64))  # Batch of 4 sequences
logits, loss = model(xb, yb)  # All 64 positions processed in parallel

# Inference (Decode): 64 forward passes for 64 tokens
context = torch.zeros((1, 1), dtype=torch.long)
for _ in range(64):
    logits, _ = model(context)
    next_token = sample(logits[:, -1, :])
    context = torch.cat([context, next_token], dim=1)
```

##### Attention Masking in Both Phases

Both phases use **causal masking** to prevent attending to future tokens:

```python
# In Head.forward() - applies to both prefill and decode
self.tril = torch.tril(torch.ones(block_size, block_size))
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

**Causal Mask** (lower triangular):
```
Position:  0  1  2  3
Token 0:  [✓  ✗  ✗  ✗]  Can only see token 0
Token 1:  [✓  ✓  ✗  ✗]  Can see tokens 0-1
Token 2:  [✓  ✓  ✓  ✗]  Can see tokens 0-2
Token 3:  [✓  ✓  ✓  ✓]  Can see tokens 0-3
```

This ensures the model learns to predict token N using only tokens 0 to N-1, which is necessary for autoregressive generation.

##### Performance Implications

**Training (Prefill)**:
- Processes 4 sequences × 64 tokens = 256 tokens per forward pass
- ~14.86% of training time (from profiling)
- Highly parallelizable on GPU/MPS

**Inference (Decode)**:
- Processes 1 token per forward pass
- 64× slower for generating 64 tokens
- Limited parallelism (sequential dependency)
- Can be optimized with KV caching (not implemented in this codebase)

##### Summary

- **`train.py` uses Prefill phase exclusively**: All tokens in a batch processed in parallel for efficient training
- **Decode phase is for inference only**: Used in `model.generate()` for autoregressive text generation
- **Key difference**: Prefill = parallel processing of known sequences, Decode = sequential generation of new tokens
- **Why it matters**: Understanding this distinction is crucial for optimizing training (prefill) vs. inference (decode) performance

### 6. Experiment Tracking

**Purpose**: Log metrics, hyperparameters, and artifacts for analysis.

**Dual Tracking System**:

#### TensorBoard (Always Enabled)
- **Metrics**: Training loss, validation loss, learning rate
- **Histograms**: Parameter distributions, gradient distributions
- **Scalars**: Per-iteration and per-evaluation metrics
- **Output**: `runs/<run_id>/logs/tensorboard/`
- **Viewing**: `tensorboard --logdir runs/<run_id>/logs/tensorboard`

#### Aim (Optional)
- **Metrics**: Same as TensorBoard with additional context
- **Distributions**: Parameter and gradient distributions
- **Metadata**: Git commit, branch, Python version
- **Configuration**: Complete hyperparameter tracking
- **Output**: `.aim/` directory (shared across runs)
- **Viewing**: `aim up`
- **Enable**: Set `aim_logging: true` in `train.yaml`

**Logging Intervals**:
- Training loss: Every iteration
- Evaluation metrics: Every `eval_interval` iterations (default: 100)
- Histograms: Every `histogram_interval` iterations (configurable)

### 7. Optimizer Configuration

**Purpose**: Update model parameters to minimize loss.

**Optimizer**: AdamW (Adam with Weight Decay)
- Adaptive learning rates per parameter
- Momentum with first and second moment estimates
- Weight decay for regularization (decoupled from gradient)

**Why AdamW**:
- Better generalization than Adam (decoupled weight decay)
- Adaptive learning rates handle different parameter scales
- Momentum helps escape local minima
- Industry standard for transformer training

**Cost**: ~27% of total training time (see `profile_model.py`)

### 8. Model Persistence

**Purpose**: Save trained model for inference and deployment.

**Format**: SafeTensors
- **Security**: No arbitrary code execution (unlike pickle)
- **Performance**: Zero-copy loading via memory mapping
- **Interoperability**: Compatible with PyTorch, Hugging Face, ONNX
- **Determinism**: Reproducible serialization

**Saved Artifacts**:
1. `model.safetensors`: Model weights in SafeTensors format
2. `config.json`: Configuration used for training
3. Metadata files: Git info, environment info, full config

**Location**: `runs/<run_id>/artifacts/export/`

## Data Flow Diagram

```
┌─────────────┐
│ CLI Args    │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────────────┐
│ Configuration Loading & Merging         │
│ • ConfigLoader (configs/)               │
│ • CLI overrides                         │
│ • ModelConfig validation                │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────┐
│ Run Setup                               │
│ • Generate Run_ID                       │
│ • Create directory structure            │
│ • Capture metadata (git, env)           │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────┐
│ Data Loading & Tokenization            │
│ • Load text file                        │
│ • Split train/val (90/10)               │
│ • Create character tokenizer            │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────┐
│ Model Initialization                    │
│ • Create LanguageModel                  │
│ • Move to device (MPS/CUDA/CPU)         │
│ • Apply torch.compile (optional)        │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────┐
│ Experiment Tracking Setup               │
│ • Initialize TensorBoard writer         │
│ • Initialize Aim tracker (optional)     │
│ • Track hyperparameters & metadata      │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────┐
│ Training Loop (max_iters iterations)    │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Periodic Evaluation             │   │
│  │ • estimate_loss() on train/val  │   │
│  │ • Log to TensorBoard & Aim      │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Training Step                   │   │
│  │ • get_batch()                   │   │
│  │ • Forward pass                  │   │
│  │ • Backward pass                 │   │
│  │ • Optimizer step                │   │
│  │ • Log training loss             │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Periodic Histogram Logging      │   │
│  │ • Log parameter distributions   │   │
│  │ • Log gradient distributions    │   │
│  └─────────────────────────────────┘   │
└──────┬──────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────┐
│ Cleanup & Persistence                   │
│ • Close Aim tracker                     │
│ • Close TensorBoard writer              │
│ • Save model.safetensors                │
│ • Save config.json                      │
│ • Print completion summary              │
└─────────────────────────────────────────┘
```

## Key Design Patterns

### 1. Configuration Hierarchy Pattern
Multiple configuration sources with clear precedence rules:
- CLI args override config files
- Config files override defaults
- Validation happens after merging

### 2. Run Isolation Pattern
Each training run is completely isolated:
- Unique Run_ID
- Separate directory structure
- Independent metadata capture
- No cross-contamination between runs

### 3. Dual Tracking Pattern
Two complementary experiment tracking systems:
- TensorBoard: Always on, lightweight, real-time visualization
- Aim: Optional, comprehensive, cross-run comparison

### 4. Try-Finally Pattern
Ensures cleanup even on failure:
```python
try:
    # Training loop
    for iter in range(max_iters):
        # ... training code ...
finally:
    # Always close trackers
    if aim_tracker is not None:
        aim_tracker.close()
    tb_writer.close()
```

### 5. Backward Compatibility Pattern
Supports both old and new configuration formats:
- New: `configs/` directory with YAML/JSON files
- Old: Single `config.json` file
- Automatic detection and fallback

## Performance Characteristics

Based on profiling results from `profile_model.py`:

| Operation | % of Total Time | Description |
|-----------|----------------|-------------|
| Backward pass | 28.57% | Gradient computation via backpropagation |
| Optimizer step | 27.32% | AdamW parameter updates |
| Forward pass | 14.86% | Model inference |
| Attention (bmm) | 6.38% | Batch matrix multiplication for attention scores |
| Linear backward | 10.38% | Gradient computation for linear layers |

**Total Training Time Breakdown**:
- Gradient computation: ~39% (backward + linear backward)
- Parameter updates: ~27% (optimizer)
- Forward inference: ~15%
- Other operations: ~19%

## Error Handling

### Configuration Errors
- Validation before training starts
- Clear error messages for invalid parameters
- Fails fast to avoid wasted computation

### Training Errors
- Try-finally ensures cleanup
- Aim tracker closed even on failure
- TensorBoard writer flushed to disk

### File System Errors
- Directory creation with error handling
- Metadata capture with fallback
- Model saving with validation

## Extension Points

### Adding New Metrics
1. Add metric to `estimate_loss()` function
2. Log to TensorBoard via `tb_logger`
3. Log to Aim via `aim_tracker` (if enabled)

### Adding New Optimizers
1. Replace `torch.optim.AdamW` with desired optimizer
2. Update configuration to include optimizer-specific parameters
3. Update documentation

### Adding Checkpointing
1. Use `output_paths['checkpoints']` directory
2. Save model state at intervals
3. Implement checkpoint loading in training loop

### Adding Learning Rate Scheduling
1. Create scheduler after optimizer
2. Call `scheduler.step()` in training loop
3. Log learning rate to TensorBoard/Aim

## Dependencies

### Core Dependencies
- `torch`: PyTorch deep learning framework
- `safetensors`: Secure model serialization

### Configuration
- `config.py`: Legacy configuration loading
- `parameters.py`: Type-safe ModelConfig dataclass
- `utils.alg.utils_config_loader`: New configuration system

### Data
- `data.py`: Data loading and batching
- `tokenizer.py`: Character-level tokenization

### Model
- `model.py`: LanguageModel architecture

### Tracking
- `tensorboard.*`: TensorBoard integration
- `utils.aim.*`: Aim experiment tracking

### Utilities
- `utils.alg.utils_run_manager`: Run management
- `utils.alg.utils_output`: Output formatting

## Usage Examples

### Basic Training
```bash
python train.py
```

### Custom Configuration
```bash
python train.py --config my_config.json
```

### Override Hyperparameters
```bash
python train.py --batch-size 8 --learning-rate 0.001 --max-iters 5000
```

### Tagged Run
```bash
python train.py --run-tag baseline
# Creates: runs/YYYYMMDD_HHMMSS_baseline/
```

### Multiple Overrides
```bash
python train.py \
    --batch-size 16 \
    --block-size 128 \
    --n-embd 256 \
    --n-head 8 \
    --n-layer 6 \
    --learning-rate 0.001 \
    --max-iters 10000 \
    --run-tag large-model
```

## Best Practices

### Configuration Management
- Use `configs/` directory for version-controlled configurations
- Use CLI args for experimental overrides
- Tag runs with descriptive names (`--run-tag`)

### Experiment Tracking
- Enable Aim for important experiments (`aim_logging: true`)
- Use TensorBoard for quick iteration and debugging
- Review histograms to detect gradient issues

### Reproducibility
- Always commit code before training
- Review `meta/git.txt` to verify clean state
- Use `meta/full_config.json` to reproduce runs

### Performance
- Enable `torch_compile: true` for faster training
- Use appropriate batch size for your hardware
- Monitor GPU/MPS utilization

### Model Persistence
- Use SafeTensors format (default)
- Save configuration alongside model
- Test loading before deploying

## Future Enhancements

### Planned Features
1. **Checkpointing**: Save model at intervals for recovery
2. **Learning Rate Scheduling**: Adaptive learning rate decay
3. **Gradient Clipping**: Prevent exploding gradients
4. **Mixed Precision Training**: FP16 for faster training
5. **Distributed Training**: Multi-GPU support
6. **Early Stopping**: Stop when validation loss plateaus
7. **Hyperparameter Search**: Automated tuning with Optuna/Ray Tune

### Architecture Improvements
1. **Plugin System**: Extensible callbacks for custom logic
2. **Configuration Validation**: JSON Schema validation
3. **Metric Registry**: Centralized metric definitions
4. **Artifact Management**: Versioned model artifacts
5. **Remote Logging**: Cloud-based experiment tracking

## Conclusion

`train.py` implements a production-ready training pipeline with:
- Flexible configuration management
- Comprehensive experiment tracking
- Reproducible run management
- Efficient training loop
- Secure model persistence

The architecture balances simplicity for quick iteration with robustness for production use, making it suitable for both research and deployment scenarios.
