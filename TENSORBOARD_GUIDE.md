# TensorBoard Integration Guide

This guide explains how to use TensorBoard to visualize and monitor your GPT model training in the ALG1 project.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [How It Works](#how-it-works)
4. [Running Training with TensorBoard](#running-training-with-tensorboard)
5. [Launching TensorBoard](#launching-tensorboard)
6. [Understanding the Metrics](#understanding-the-metrics)
7. [Comparing Multiple Runs](#comparing-multiple-runs)
8. [Troubleshooting](#troubleshooting)

---

## Overview

TensorBoard is automatically integrated into the training pipeline. Every time you run `train.py`, training metrics are logged to TensorBoard format, allowing you to:

- Monitor training and validation loss in real-time
- Track learning rate changes
- Compare multiple training runs
- Identify overfitting or training issues early

**No configuration needed** - TensorBoard logging is enabled by default.

---

## Prerequisites

TensorBoard is already included in the project dependencies. If you need to install it manually:

```bash
uv pip install "tensorboard>=2.11.0"
```

---

## How It Works

### Automatic Logging

The training script automatically logs metrics at key points:

1. **Training Loss** - Logged at every iteration
   - Metric name: `Loss/train_step`
   - Frequency: Every training step
   - Shows immediate feedback on model learning

2. **Evaluation Metrics** - Logged at evaluation intervals
   - `Loss/train` - Average training loss over evaluation batches
   - `Loss/val` - Average validation loss over evaluation batches
   - `Learning_Rate` - Current learning rate
   - Frequency: Every `eval_interval` iterations (default: 500)

### Directory Structure

TensorBoard logs are saved in your run directory:

```
runs/
└── 2026-02-09_001/              # Your run ID
    └── logs/
        └── tensorboard/
            └── events.out.tfevents.[timestamp].[hostname]
```

Each training run gets its own unique directory, making it easy to compare runs later.

---

## Running Training with TensorBoard

### Basic Training

Simply run your training as usual:

```bash
# Activate the environment
source .venv/bin/activate

# Run training with default settings
python train.py
```

### Custom Training Parameters

Override parameters as needed:

```bash
# Short training run for testing
python train.py --max-iters 100 --eval-interval 20

# Longer training with custom config
python train.py --config configs/my_config.json --max-iters 10000
```

### What You'll See

During training, you'll see normal output plus TensorBoard confirmation:

```
=== Training Run: 2026-02-09_001 ===
...
step 0: train loss 4.3612, val loss 4.3589
step 500: train loss 2.1234, val loss 2.3456
...
Training complete! Saving model weights...

TensorBoard logs saved to: /path/to/runs/2026-02-09_001/logs/tensorboard
View with: tensorboard --logdir=/path/to/runs/2026-02-09_001/logs/tensorboard
Or open folder: file:///path/to/runs/2026-02-09_001/logs/tensorboard
```

---

## Launching TensorBoard

### Option 1: View Single Run

After training completes, use the command printed in the output:

```bash
tensorboard --logdir=runs/2026-02-09_001/logs/tensorboard
```

### Option 2: View All Runs (Recommended)

To compare multiple training runs, point TensorBoard to the parent directory:

```bash
tensorboard --logdir=runs
```

This will load all runs and allow side-by-side comparison.

### Option 3: Specific Port

If port 6006 is already in use:

```bash
tensorboard --logdir=runs --port=6007
```

### Accessing the Web UI

Once TensorBoard starts, you'll see:

```
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

Open your browser and navigate to: **http://localhost:6006/**

---

## Understanding the Metrics

### Scalars Tab

This is where you'll spend most of your time. Key metrics to monitor:

#### 1. Loss/train_step
- **What it shows**: Training loss at every iteration
- **What to look for**: 
  - Should decrease over time
  - Noisy but trending downward = healthy training
  - Flat line = model not learning
  - Increasing = learning rate too high or other issues

#### 2. Loss/train
- **What it shows**: Average training loss at evaluation intervals
- **What to look for**:
  - Smoother than train_step (averaged over multiple batches)
  - Should steadily decrease
  - Compare with validation loss to detect overfitting

#### 3. Loss/val
- **What it shows**: Validation loss at evaluation intervals
- **What to look for**:
  - Should decrease along with training loss
  - If val loss increases while train loss decreases = **overfitting**
  - Gap between train and val loss = generalization gap

#### 4. Learning_Rate
- **What it shows**: Current learning rate
- **What to look for**:
  - Constant in this implementation (no scheduler)
  - Useful when comparing runs with different learning rates

### Smoothing

Use the smoothing slider (left sidebar) to reduce noise in the plots:
- 0.0 = raw data (noisy)
- 0.6 = default (good balance)
- 0.9 = very smooth (may hide important details)

### Time vs Steps

Toggle between:
- **STEP** (default): X-axis shows iteration number
- **RELATIVE**: X-axis shows time since start
- **WALL**: X-axis shows absolute time

---

## Comparing Multiple Runs

### Running Multiple Experiments

```bash
# Run 1: Baseline
python train.py --max-iters 5000

# Run 2: Higher learning rate
python train.py --max-iters 5000 --learning-rate 0.001

# Run 3: Different batch size
python train.py --max-iters 5000 --batch-size 128
```

### Viewing Comparisons

1. Launch TensorBoard with all runs:
   ```bash
   tensorboard --logdir=runs
   ```

2. In the web UI:
   - All runs appear in the left sidebar
   - Each run has a different color
   - Toggle runs on/off by clicking the eye icon
   - Hover over lines to see exact values

3. Use the "Runs" selector to:
   - Show/hide specific runs
   - Filter by run name or tag
   - Compare metrics side-by-side

### Best Practices for Comparison

- **Consistent naming**: Use descriptive run IDs or add notes in `runs/[run-id]/meta/notes.md`
- **Same scale**: Ensure you're comparing runs with similar iteration counts
- **Focus on validation loss**: Training loss can be misleading; validation loss shows true performance

---

## Troubleshooting

### TensorBoard Not Available

If you see: `Warning: TensorBoard not available. Install with: pip install tensorboard`

**Solution**:
```bash
source .venv/bin/activate
uv pip install "tensorboard>=2.11.0"
```

### No Metrics Showing

**Possible causes**:
1. Training hasn't reached first evaluation interval yet
   - Wait for `eval_interval` iterations (default: 500)
   - Or run with `--eval-interval 10` for faster feedback

2. TensorBoard pointing to wrong directory
   - Check the path printed after training
   - Ensure you're using the correct `--logdir` path

3. Browser cache issue
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Or try incognito/private browsing mode

### Port Already in Use

If you see: `ERROR: TensorBoard could not bind to port 6006`

**Solution**:
```bash
# Use a different port
tensorboard --logdir=runs --port=6007

# Or kill the existing TensorBoard process
pkill -f tensorboard
```

### Training Continues Without TensorBoard

This is **expected behavior**. If TensorBoard fails to initialize:
- Training continues normally
- A warning is printed
- No metrics are logged
- Model training is unaffected

This ensures TensorBoard issues don't break your training runs.

### Event Files Not Created

**Check**:
1. Run directory exists: `ls runs/`
2. Logs directory exists: `ls runs/[run-id]/logs/tensorboard/`
3. TensorBoard initialized successfully (no warnings during training)

---

## Tips and Best Practices

### 1. Monitor Training in Real-Time

Launch TensorBoard before or during training:

```bash
# Terminal 1: Start TensorBoard
tensorboard --logdir=runs

# Terminal 2: Run training
source .venv/bin/activate
python train.py
```

TensorBoard will automatically detect new data and update the plots.

### 2. Use Evaluation Intervals Wisely

- **Short runs** (< 1000 iters): Use `--eval-interval 50` for frequent updates
- **Long runs** (> 10000 iters): Use `--eval-interval 500` to reduce overhead
- **Production**: Balance between monitoring frequency and training speed

### 3. Save Important Runs

Keep successful runs for future reference:

```bash
# Copy a good run to a permanent location
cp -r runs/2026-02-09_001 saved_models/baseline_v1
```

### 4. Clean Up Old Runs

TensorBoard logs are small, but can accumulate:

```bash
# Remove old runs (be careful!)
rm -rf runs/2026-02-08_*

# Or archive them
tar -czf old_runs_2026-02-08.tar.gz runs/2026-02-08_*
rm -rf runs/2026-02-08_*
```

---

## Advanced Usage

### Filtering Runs by Regex

```bash
# Only show runs from February 9th
tensorboard --logdir=runs --path_prefix="2026-02-09"
```

### Exporting Data

In TensorBoard web UI:
1. Click the download icon (⬇) in the top right
2. Choose CSV format
3. Use for custom analysis or plotting

### Remote Access

If training on a remote server:

```bash
# On remote server
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@remote-server
```

Then access at http://localhost:6006/ on your local machine.

---

## Summary

TensorBoard integration is automatic and requires no configuration. Simply:

1. **Run training**: `python train.py`
2. **Launch TensorBoard**: `tensorboard --logdir=runs`
3. **Open browser**: http://localhost:6006/
4. **Monitor metrics**: Watch loss curves and learning progress

For questions or issues, refer to the [TensorBoard documentation](https://www.tensorflow.org/tensorboard) or check the troubleshooting section above.
