# Quick Start Guide

## Setup (One Time)

```bash
# Activate virtual environment
source .venv/bin/activate
```

## Training

```bash
# Quick test (10 iterations)
python train.py --max-iters 10

# Full training with optimized settings
python train.py

# Custom training
python train.py --max-iters 5000 --batch-size 64 --learning-rate 0.003
```

## Inference

```bash
# Interactive mode (will prompt for input)
python inference.py

# With specific prompt
python inference.py --prompt "ROMEO:" --max_tokens 200

# From specific training run
python inference.py --run-id "2026-02-13_003" --prompt "Hello" --max_tokens 500
```

## Current Configuration

**Model**: 10.8M parameters
- Embedding dimension: 384
- Attention heads: 6
- Transformer layers: 6
- Dropout: 0.2

**Training**:
- Batch size: 64
- Context length: 256 tokens
- Learning rate: 0.003
- Max iterations: 5000

**Logging**:
- TensorBoard: Always enabled
- Aim: Configurable (default: disabled)
  - Edit `configs/train.yaml` to enable: `aim_logging: true`
  - See `AIM_CONFIGURATION.md` for details

**Device**: MPS (Mac M4 GPU acceleration) ✅

## Verification

```bash
# Check PyTorch and MPS status
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

Expected output:
```
PyTorch: 2.10.0
MPS: True
```

## Training Outputs

Each training run creates a directory in `runs/` with:
- `artifacts/export/model.safetensors` - Trained model weights
- `artifacts/export/config.json` - Configuration used
- `logs/tensorboard/` - TensorBoard logs
- `meta/` - Git info, environment, etc.

## View Training Progress

```bash
# Launch TensorBoard
tensorboard --logdir=runs --port=6006

# Then open: http://localhost:6006/
```

## Status: ✅ All Working

- ✅ Training with MPS acceleration
- ✅ Inference with trained models
- ✅ Optimized for Mac M4
- ✅ SafeTensors model format
- ✅ Run management and tracking
