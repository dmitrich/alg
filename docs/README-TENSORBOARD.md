# TensorBoard Visualization Guide

This guide explains how to use TensorBoard to visualize and monitor your machine learning training runs.

## Table of Contents

- [What is TensorBoard?](#what-is-tensorboard)
- [Quick Start](#quick-start)
- [Launching TensorBoard](#launching-tensorboard)
- [Navigating TensorBoard](#navigating-tensorboard)
- [Viewing Metrics](#viewing-metrics)
- [Exploring Histograms](#exploring-histograms)
- [Comparing Multiple Runs](#comparing-multiple-runs)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## What is TensorBoard?

TensorBoard is TensorFlow's visualization toolkit that helps you:
- Monitor training and validation metrics in real-time
- Visualize model architecture and parameter distributions
- Compare multiple training runs
- Debug training issues with histogram visualizations
- Track hyperparameters and their impact on performance

All TensorBoard logs are stored in the `runs/<run_id>/logs/tensorboard/` directory.

---

## Quick Start

### 1. Run Training

Training automatically logs metrics to TensorBoard:

```bash
python train.py
```

### 2. Launch TensorBoard

After training starts (or completes), launch TensorBoard:

```bash
tensorboard --logdir=runs/2026-02-10_012/logs/tensorboard --port=6006
```

Or view all runs at once:

```bash
tensorboard --logdir=runs --port=6006
```

### 3. Open in Browser

Navigate to: http://localhost:6006

---

## Launching TensorBoard

### View a Single Run

To view a specific training run:

```bash
tensorboard --logdir=runs/2026-02-10_012/logs/tensorboard --port=6006
```

Replace `2026-02-10_012` with your run ID.

### View All Runs

To compare all training runs:

```bash
tensorboard --logdir=runs --port=6006
```

This loads all runs from the `runs/` directory, allowing side-by-side comparison.

### Custom Port

If port 6006 is already in use:

```bash
tensorboard --logdir=runs --port=6007
```

Then open: http://localhost:6007

### Bind to All Network Interfaces

To access TensorBoard from other machines on your network:

```bash
tensorboard --logdir=runs --host=0.0.0.0 --port=6006
```

Then access from any machine: http://<your-ip>:6006

### Run in Background

To keep TensorBoard running in the background:

```bash
tensorboard --logdir=runs --port=6006 &
```

Or use `nohup` to keep it running after logout:

```bash
nohup tensorboard --logdir=runs --port=6006 > tensorboard.log 2>&1 &
```

### Stop TensorBoard

To stop a running TensorBoard server:

```bash
# Find the process
ps aux | grep tensorboard

# Kill the process
kill <PID>
```

Or use Ctrl+C in the terminal where it's running.

---

## Navigating TensorBoard

### Main Tabs

TensorBoard organizes visualizations into several tabs:

1. **SCALARS** - Training and validation metrics over time
2. **GRAPHS** - Model architecture visualization (if available)
3. **DISTRIBUTIONS** - Parameter and gradient distributions
4. **HISTOGRAMS** - 3D visualization of distributions over time
5. **TIME SERIES** - Custom time series data

### Scalars Tab

The Scalars tab is your primary tool for monitoring training progress.

**Available Metrics:**
- `train/loss` - Training loss at each iteration
- `eval/train_loss` - Training loss during evaluation
- `eval/val_loss` - Validation loss during evaluation
- `eval/learning_rate` - Learning rate schedule

**Key Features:**
- Smooth curves with adjusting slider
- Toggle logarithmic scale
- Download data as CSV or JSON
- Zoom and pan on charts

---

## Viewing Metrics

### Training Loss

**Location:** Scalars → train/loss

This shows the loss computed at every training iteration. It's typically noisy because it's computed on individual batches.

**How to interpret:**
- Should generally trend downward
- Noise is normal (batch-to-batch variation)
- Use smoothing slider to see overall trend
- Sudden spikes may indicate learning rate issues

**Smoothing:**
1. Adjust the "Smoothing" slider (top left)
2. Recommended: 0.6-0.8 for training loss
3. Higher values = smoother curve, but may hide important details

### Evaluation Metrics

**Location:** Scalars → eval/

These metrics are computed periodically (every `eval_interval` iterations) on multiple batches for more stable estimates.

**eval/train_loss:**
- Training loss averaged over `eval_iters` batches
- Less noisy than train/loss
- Better for comparing runs

**eval/val_loss:**
- Validation loss averaged over `eval_iters` batches
- Most important metric for model performance
- Should decrease during training
- If it increases while train_loss decreases → overfitting

**eval/learning_rate:**
- Shows learning rate schedule
- Useful if using learning rate decay
- Currently constant in this project

### Comparing Train vs Validation Loss

**To detect overfitting:**

1. Go to Scalars tab
2. Look at eval/train_loss and eval/val_loss
3. If val_loss stops decreasing while train_loss continues → overfitting
4. If both decrease together → healthy training

**Healthy training pattern:**
```
train_loss: ↓↓↓↓↓↓
val_loss:   ↓↓↓↓↓↓
```

**Overfitting pattern:**
```
train_loss: ↓↓↓↓↓↓
val_loss:   ↓↓↓→→↑
```

---

## Exploring Histograms

### Enabling Histogram Logging

Histograms are controlled by the `histogram_interval` setting in `configs/tensorboard.json`.

**To enable:**
```json
{
  "histogram_interval": 100
}
```

**To disable (faster training):**
```json
{
  "histogram_interval": 0
}
```

### Viewing Parameter Distributions

**Location:** Distributions or Histograms tab

When histogram logging is enabled, you can view:
- Parameter distributions (weights and biases)
- Gradient distributions
- How they evolve during training

**Available distributions:**
- `transformer.h.0.attn.c_attn.weight` - Attention weights
- `transformer.h.0.attn.c_attn.bias` - Attention biases
- `transformer.h.0.mlp.c_fc.weight` - MLP weights
- And many more...

### Interpreting Histograms

**Healthy parameter distributions:**
- Roughly Gaussian (bell-shaped)
- Centered around zero (for weights)
- Stable over time (not exploding or vanishing)

**Warning signs:**
- All values near zero → vanishing gradients
- Very large values → exploding gradients
- Bimodal distributions → potential training instability

**Healthy gradient distributions:**
- Centered around zero
- Consistent scale across layers
- Not too small (vanishing) or too large (exploding)

### Histogram Tab vs Distributions Tab

**Distributions Tab:**
- 2D line plots showing distribution quantiles
- Easier to see overall trends
- Better for comparing across time

**Histograms Tab:**
- 3D visualization with time on one axis
- Shows full distribution shape
- More detailed but harder to read

---

## Comparing Multiple Runs

### Loading Multiple Runs

To compare different experiments:

```bash
tensorboard --logdir=runs --port=6006
```

TensorBoard will automatically detect all runs in subdirectories.

### Run Selection

**In the TensorBoard UI:**
1. Look at the left sidebar
2. Each run appears with its directory name
3. Toggle runs on/off using checkboxes
4. Different runs appear in different colors

### Comparing Hyperparameters

**Example: Compare different learning rates**

1. Run multiple experiments:
   ```bash
   python train.py --learning-rate 0.01 --run-tag lr-001
   python train.py --learning-rate 0.02 --run-tag lr-002
   python train.py --learning-rate 0.05 --run-tag lr-005
   ```

2. Launch TensorBoard:
   ```bash
   tensorboard --logdir=runs --port=6006
   ```

3. View eval/val_loss for all runs
4. Identify which learning rate performs best

### Filtering Runs

Use the regex filter in the left sidebar to show/hide specific runs:

**Show only runs from today:**
```
2026-02-10.*
```

**Show only baseline runs:**
```
.*baseline.*
```

**Exclude test runs:**
```
^(?!.*test).*
```

---

## Advanced Features

### Downloading Data

**Export scalar data:**
1. Hover over any chart
2. Click the download icon (bottom right)
3. Choose format: CSV or JSON
4. Data includes all points for that metric

**Use cases:**
- Create custom visualizations
- Statistical analysis in Python/R
- Include in papers or reports

### Smoothing Algorithms

TensorBoard offers different smoothing algorithms:

**Exponential Moving Average (default):**
- Smoothing slider: 0.0 to 1.0
- Higher values = more smoothing
- Good for general use

**Tips:**
- Use 0.6-0.8 for noisy training loss
- Use 0.0-0.3 for evaluation metrics (already smooth)
- Adjust based on your preference

### Relative vs Absolute Time

**Toggle between:**
- **Wall time** - Actual clock time
- **Relative time** - Time since training started
- **Step** - Training iteration number (default)

**How to change:**
1. Click the settings icon (gear) in top right
2. Select "Horizontal Axis"
3. Choose: STEP, RELATIVE, or WALL

### Logarithmic Scale

For metrics that span multiple orders of magnitude:

1. Hover over a chart
2. Click the settings icon
3. Toggle "Y-axis: log scale"

Useful for:
- Loss values that start very high
- Learning rate schedules with decay

---

## Troubleshooting

### Issue: "No dashboards are active"

**Problem:** TensorBoard shows no data.

**Solution:**
1. Verify training has started and logged some data
2. Check that log directory exists:
   ```bash
   ls -la runs/2026-02-10_012/logs/tensorboard/
   ```
3. Verify event files exist:
   ```bash
   ls -la runs/2026-02-10_012/logs/tensorboard/*.tfevents.*
   ```
4. Try refreshing the browser (Ctrl+R or Cmd+R)
5. Restart TensorBoard with correct path

### Issue: Port already in use

**Problem:** Error: "Address already in use"

**Solution:**
```bash
# Use a different port
tensorboard --logdir=runs --port=6007
```

Or kill the existing TensorBoard process:
```bash
# Find the process
lsof -i :6006

# Kill it
kill <PID>
```

### Issue: Runs not appearing

**Problem:** Some runs don't show up in TensorBoard.

**Solution:**
1. Verify the run directory structure:
   ```bash
   ls -la runs/
   ```
2. Check that each run has a tensorboard directory:
   ```bash
   ls -la runs/*/logs/tensorboard/
   ```
3. Restart TensorBoard:
   ```bash
   # Stop current instance (Ctrl+C)
   # Restart
   tensorboard --logdir=runs --port=6006
   ```
4. Clear TensorBoard cache:
   ```bash
   rm -rf /tmp/.tensorboard-info/
   tensorboard --logdir=runs --port=6006
   ```

### Issue: Metrics not updating in real-time

**Problem:** Charts don't show latest data during training.

**Solution:**
- TensorBoard auto-refreshes every 30 seconds
- Manually refresh: Click the refresh icon (top right)
- Or press Ctrl+R (Cmd+R on Mac)
- Ensure TensorBoard is pointing to the correct directory

### Issue: Histograms not appearing

**Problem:** Distributions/Histograms tabs are empty.

**Solution:**
1. Check if histogram logging is enabled:
   ```bash
   cat configs/tensorboard.json
   ```
2. Verify `histogram_interval` is > 0
3. Histograms are only logged at specified intervals
4. Wait for training to reach a histogram checkpoint
5. Refresh TensorBoard

### Issue: Out of memory when loading many runs

**Problem:** TensorBoard crashes or becomes slow with many runs.

**Solution:**
1. Load fewer runs at once:
   ```bash
   # Load only recent runs
   tensorboard --logdir=runs/2026-02-10* --port=6006
   ```
2. Increase memory limit:
   ```bash
   tensorboard --logdir=runs --port=6006 --max_reload_threads=1
   ```
3. Archive old runs:
   ```bash
   mkdir runs_archive
   mv runs/2026-01-* runs_archive/
   ```

---

## Configuration

### TensorBoard Configuration File

Location: `configs/tensorboard.json`

```json
{
  "runs_dir": "runs",
  "port": 6006,
  "histogram_interval": 100
}
```

**Settings:**
- `runs_dir` - Base directory for all runs (default: "runs")
- `port` - Default port for TensorBoard server (default: 6006)
- `histogram_interval` - How often to log histograms (default: 100)
  - Set to 0 to disable histogram logging
  - Higher values = less frequent logging = faster training

### Histogram Interval Recommendations

**For development/debugging:**
```json
{
  "histogram_interval": 50
}
```
- Frequent updates
- Slower training
- Good for monitoring gradient flow

**For production training:**
```json
{
  "histogram_interval": 500
}
```
- Less frequent updates
- Faster training
- Still captures important trends

**To disable (fastest training):**
```json
{
  "histogram_interval": 0
}
```

---

## Best Practices

### During Training

**Monitor these metrics:**
1. **train/loss** - Should decrease (with noise)
2. **eval/val_loss** - Should decrease smoothly
3. **Gap between train and val loss** - Should be small

**Warning signs:**
- Val loss increasing → overfitting or learning rate too high
- Loss not decreasing → learning rate too low or bad initialization
- Loss = NaN → exploding gradients or numerical instability

### Organizing Experiments

**Use descriptive run tags:**
```bash
python train.py --run-tag baseline-lr002
python train.py --run-tag deep-8layer
python train.py --run-tag high-dropout
```

**Benefits:**
- Easy to identify runs in TensorBoard
- Clear experiment organization
- Better for team collaboration

### Comparing Experiments

**Best practices:**
1. Change one variable at a time
2. Run multiple seeds for statistical significance
3. Use consistent evaluation intervals
4. Document changes in run tags or notes

### Cleaning Up

**Archive old runs:**
```bash
# Create archive directory
mkdir runs_archive

# Move old runs
mv runs/2026-01-* runs_archive/

# Or compress
tar -czf runs_archive_2026-01.tar.gz runs/2026-01-*
rm -rf runs/2026-01-*
```

---

## Integration with Aim

This project uses both TensorBoard and Aim for experiment tracking:

**TensorBoard strengths:**
- Real-time monitoring during training
- Detailed histogram visualizations
- Familiar interface for TensorFlow users

**Aim strengths:**
- Better run comparison and filtering
- More flexible querying
- Hyperparameter exploration
- Persistent storage

**Recommended workflow:**
1. Use TensorBoard during training for real-time monitoring
2. Use Aim after training for analysis and comparison
3. Both tools track the same metrics automatically

See [README-AIM.md](README-AIM.md) for Aim usage instructions.

---

## Additional Resources

- **TensorBoard Documentation:** https://www.tensorflow.org/tensorboard
- **TensorBoard GitHub:** https://github.com/tensorflow/tensorboard
- **TensorBoard Tutorial:** https://www.tensorflow.org/tensorboard/get_started

---

## Quick Reference

### Common Commands

```bash
# View single run
tensorboard --logdir=runs/2026-02-10_012/logs/tensorboard --port=6006

# View all runs
tensorboard --logdir=runs --port=6006

# Custom port
tensorboard --logdir=runs --port=6007

# Bind to all interfaces
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# Run in background
tensorboard --logdir=runs --port=6006 &

# With nohup
nohup tensorboard --logdir=runs --port=6006 > tensorboard.log 2>&1 &
```

### Keyboard Shortcuts

- `Ctrl+R` (Cmd+R on Mac) - Refresh page
- `Ctrl+F` (Cmd+F on Mac) - Search/filter runs
- `Esc` - Close modals

### URL Parameters

```
# Open specific tab
http://localhost:6006/#scalars

# Filter runs
http://localhost:6006/#scalars&regexInput=baseline

# Smooth curves
http://localhost:6006/#scalars&smoothing=0.8
```

---

**Happy Training! 📊**
