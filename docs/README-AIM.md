# Aim Experiment Tracking Guide

This guide explains how to use Aim to track, visualize, and compare your machine learning experiments.

## Table of Contents

- [What is Aim?](#what-is-aim)
- [Quick Start](#quick-start)
- [Launching the Aim UI](#launching-the-aim-ui)
- [Navigating the Aim UI](#navigating-the-aim-ui)
- [Filtering and Comparing Runs](#filtering-and-comparing-runs)
- [Viewing Metrics](#viewing-metrics)
- [Exploring Hyperparameters](#exploring-hyperparameters)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## What is Aim?

Aim is an open-source experiment tracking tool that helps you:
- Track metrics, hyperparameters, and metadata across training runs
- Compare multiple experiments side-by-side
- Filter runs by hyperparameters or metadata
- Visualize training curves and distributions
- Organize experiments with tags and contexts

All experiment data is stored in the `.aim` directory in your project root.

---

## Quick Start

### 1. Run Training

Simply run your training script as usual. Aim tracking is automatically enabled:

```bash
python train.py
```

Or with custom parameters:

```bash
python train.py --max-iters 5000 --learning-rate 0.01 --run-tag baseline
```

### 2. Launch Aim UI

After training completes, launch the Aim UI:

```bash
aim up
```

### 3. Open in Browser

Navigate to the URL shown in your terminal (default: http://localhost:43800)

---

## Launching the Aim UI

### Basic Launch

```bash
aim up
```

This starts the Aim server on the default port (43800) and opens your browser automatically.

### Custom Port

If port 43800 is already in use:

```bash
aim up --port 6007
```

Then open: http://localhost:6007

### Specify Repository Path

If your `.aim` directory is in a different location:

```bash
aim up --repo /path/to/your/.aim
```

### Run in Background

To keep the server running in the background:

```bash
aim up --host 0.0.0.0 --port 43800 &
```

### Stop the Server

To stop the Aim server:

```bash
# Find the process
ps aux | grep "aim up"

# Kill the process
kill <PID>
```

Or use Ctrl+C in the terminal where it's running.

---

## Navigating the Aim UI

### Main Sections

1. **Metrics Explorer** - Compare and visualize metrics across runs
2. **Runs Table** - View all runs with their hyperparameters and metadata
3. **Params** - Explore hyperparameter distributions
4. **Images** - View tracked images (if any)
5. **Distributions** - Explore parameter and gradient distributions

### Metrics Explorer

The Metrics Explorer is your primary tool for comparing training runs.

**Key Features:**
- Plot multiple metrics on the same chart
- Group runs by hyperparameters
- Apply smoothing to noisy curves
- Zoom and pan on charts
- Export charts as images

**How to Use:**
1. Click "Metrics" in the left sidebar
2. Select metrics to visualize (e.g., "loss")
3. Use the context selector to filter by subset (train/val)
4. Group runs by clicking "Group by" and selecting a hyperparameter

### Runs Table

View all your experiments in a sortable, filterable table.

**Columns Include:**
- Run name (experiment ID)
- Hyperparameters (learning_rate, batch_size, etc.)
- Metadata (git_commit, python_version, etc.)
- Run directory path
- Creation date

**How to Use:**
1. Click "Runs" in the left sidebar
2. Sort by any column (click column header)
3. Filter runs using the search bar
4. Click a run to view detailed information

---

## Filtering and Comparing Runs

### Basic Filtering

Use the search bar to filter runs by any tracked parameter:

```
run.learning_rate > 0.01
```

```
run.n_layer == 4
```

```
run.experiment.startswith("2026-02-10")
```

### Multiple Conditions

Combine filters with `and` / `or`:

```
run.learning_rate > 0.01 and run.n_layer == 4
```

```
run.batch_size == 4 or run.batch_size == 8
```

### Common Filter Examples

**Find runs with specific learning rate:**
```
run.learning_rate == 0.02
```

**Find runs from today:**
```
run.experiment.startswith("2026-02-10")
```

**Find runs with high learning rates:**
```
run.learning_rate >= 0.05
```

**Find runs with specific tags:**
```
"baseline" in run.experiment
```

**Exclude test runs:**
```
not run.experiment.startswith("test")
```

---

## Viewing Metrics

### Available Metrics

Your training tracks the following metrics:

- **loss** - Training and validation loss
  - Context: `{'subset': 'train'}` or `{'subset': 'val'}`
  - Tracked at every iteration (train) or eval_interval (val)

### Metric Contexts

Metrics are organized by context to separate train/val data:

**To view training loss only:**
1. Select "loss" metric
2. Filter by context: `context.subset == "train"`

**To view validation loss only:**
1. Select "loss" metric
2. Filter by context: `context.subset == "val"`

**To compare train vs val:**
1. Select "loss" metric
2. Group by: `context.subset`
3. Both curves will appear on the same chart

### Smoothing Noisy Curves

Training curves can be noisy. Apply smoothing:

1. Click the "Smoothing" slider in the chart controls
2. Adjust from 0 (no smoothing) to 1 (maximum smoothing)
3. Recommended: 0.3-0.5 for most cases

---

## Exploring Hyperparameters

### View All Hyperparameters

Your runs track these hyperparameters automatically:

**Model Architecture:**
- `n_embd` - Embedding dimension
- `n_head` - Number of attention heads
- `n_layer` - Number of transformer layers
- `dropout` - Dropout rate

**Training Configuration:**
- `batch_size` - Batch size
- `block_size` - Context length
- `learning_rate` - Learning rate
- `max_iters` - Maximum iterations
- `eval_interval` - Evaluation frequency
- `eval_iters` - Evaluation iterations

**Data Configuration:**
- `data_path` - Path to training data
- `train_split` - Training split ratio
- `val_split` - Validation split ratio

### Compare Hyperparameter Impact

**Example: Compare different learning rates**

1. Go to Metrics Explorer
2. Select "loss" metric
3. Group by: `learning_rate`
4. Each learning rate will have a different color
5. Hover over curves to see exact values

**Example: Find best n_layer configuration**

1. Go to Runs Table
2. Sort by final validation loss
3. Look at `n_layer` column
4. Identify which layer count performs best

---

## Advanced Features

### Viewing Distributions

If histogram logging is enabled (`histogram_interval > 0`), you can view parameter and gradient distributions:

1. Click "Distributions" in the left sidebar
2. Select a parameter (e.g., "transformer.h.0.attn.c_attn.weight")
3. View how the distribution evolves over training
4. Filter by context: `{'type': 'param'}` or `{'type': 'grad'}`

### Exporting Data

**Export runs table to CSV:**
1. Go to Runs Table
2. Click the export icon (top right)
3. Choose CSV format
4. Save to your desired location

**Export charts as images:**
1. In Metrics Explorer, hover over a chart
2. Click the camera icon
3. Save as PNG

### Grouping Strategies

**Group by single parameter:**
```
Group by: learning_rate
```

**Group by multiple parameters:**
```
Group by: learning_rate, n_layer
```

**Group by date:**
```
Group by: run.experiment[:10]  # Groups by YYYY-MM-DD
```

---

## Troubleshooting

### Issue: "Cannot find repository"

**Problem:** Aim can't find the `.aim` directory.

**Solution:**
```bash
# Make sure you're in the project root directory
cd /path/to/your/project

# Verify .aim exists
ls -la .aim

# Launch Aim from project root
aim up
```

### Issue: Port already in use

**Problem:** Port 43800 is already occupied.

**Solution:**
```bash
# Use a different port
aim up --port 6007
```

### Issue: No runs appear in UI

**Problem:** Runs table is empty.

**Solution:**
1. Verify training completed successfully
2. Check that `.aim` directory exists and has content:
   ```bash
   ls -la .aim
   ```
3. Try refreshing the browser (Ctrl+R or Cmd+R)
4. Restart the Aim server:
   ```bash
   # Stop current server (Ctrl+C)
   # Restart
   aim up
   ```

### Issue: RocksDB Error (No such file or directory)

**Problem:** Aim tries to write to a temporary directory that was deleted.

**Solution:**

The current implementation uses `.aim` in the project root, which should work correctly. If you still encounter this error:

1. Ensure you're running training from the project root directory
2. Check that you have write permissions in the project directory
3. Verify the `.aim` directory is not on a network drive or temporary filesystem

**Alternative:** Use an absolute path for the Aim repository:

Edit `train.py` line 180:
```python
# Change from:
aim_tracker = AimTracker(run_id, run_dir)

# To:
aim_tracker = AimTracker(run_id, run_dir, repo_path='/Users/dmitri/aim_logs/project')
```

Then create the directory:
```bash
mkdir -p /Users/dmitri/aim_logs/project
```

### Issue: Metrics not updating in real-time

**Problem:** UI doesn't show latest metrics during training.

**Solution:**
- Aim UI updates when you refresh the page
- Click the refresh button in the UI
- Or press Ctrl+R (Cmd+R on Mac)
- Real-time updates require the Aim server to be running during training

### Issue: Git metadata shows "unavailable"

**Problem:** Git commit and branch show as "unavailable".

**Solution:**
1. Ensure you're in a git repository:
   ```bash
   git status
   ```
2. If not a git repo, initialize one:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
3. Git metadata will be tracked in future runs

---

## Best Practices

### Organizing Experiments

**Use run tags for experiment groups:**
```bash
python train.py --run-tag baseline
python train.py --run-tag high-lr --learning-rate 0.1
python train.py --run-tag deep --n-layer 8
```

**Consistent naming conventions:**
- Use descriptive tags: `baseline`, `high-lr`, `deep-model`
- Avoid generic tags: `test1`, `test2`, `final`

### Comparing Experiments

**When comparing runs:**
1. Keep one variable constant while changing another
2. Run multiple seeds for statistical significance
3. Use grouping to visualize trends
4. Export results for presentations

### Cleaning Up Old Runs

**To remove old experiments:**

⚠️ **Warning:** This permanently deletes experiment data!

```bash
# Backup first
cp -r .aim .aim.backup

# Remove specific runs (use Aim UI to identify hashes)
aim runs rm <run_hash>

# Or remove all runs (nuclear option)
rm -rf .aim
```

---

## Additional Resources

- **Aim Documentation:** https://aimstack.readthedocs.io/
- **Aim GitHub:** https://github.com/aimhubio/aim
- **Aim Community:** https://community.aimstack.io/

---

## Quick Reference

### Common Commands

```bash
# Launch Aim UI
aim up

# Launch on custom port
aim up --port 6007

# List all runs
aim runs ls

# Remove a specific run
aim runs rm <run_hash>

# Show Aim version
aim version
```

### Common Filters

```python
# Learning rate filters
run.learning_rate > 0.01
run.learning_rate == 0.02
run.learning_rate in [0.01, 0.02, 0.05]

# Date filters
run.experiment.startswith("2026-02-10")

# Tag filters
"baseline" in run.experiment

# Multiple conditions
run.learning_rate > 0.01 and run.n_layer == 4
```

### Keyboard Shortcuts (in UI)

- `Ctrl+K` (Cmd+K on Mac) - Open command palette
- `Ctrl+R` (Cmd+R on Mac) - Refresh page
- `Esc` - Close modals/dialogs

---

**Happy Experimenting! 🚀**
