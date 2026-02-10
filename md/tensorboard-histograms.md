# TensorBoard Histogram Logging

## Overview

The training script now supports logging histograms of model parameters and gradients to TensorBoard. This feature helps visualize the distribution of weights and gradients during training, which is useful for:

- Detecting vanishing or exploding gradients
- Monitoring weight initialization
- Understanding how parameters evolve during training
- Debugging training instability

## Configuration

Histogram logging is controlled by the `histogram_interval` parameter in `configs/tensorboard.json`:

```json
{
  "runs_directory": "runs",
  "default_port": 6006,
  "max_port_attempts": 10,
  "histogram_interval": 100
}
```

### Parameters

- **histogram_interval**: Number of training steps between histogram logging (default: 100)
  - Set to a higher value (e.g., 500) to reduce overhead and file size
  - Set to a lower value (e.g., 50) for more frequent snapshots
  - Set to match `eval_interval` to align with evaluation metrics
  - **Set to 0 or negative to disable histogram logging completely**

## What Gets Logged

### Model Parameters
For each trainable parameter in the model:
- **Parameters/{layer_name}**: Distribution of weight values
- **Gradients/{layer_name}**: Distribution of gradient values (if gradients exist)

Example parameter names:
- `Parameters/token_embedding_table.weight`
- `Parameters/blocks.0.sa.key.weight`
- `Parameters/blocks.0.ffwd.net.0.weight`
- `Parameters/lm_head.weight`

### Viewing in TensorBoard

1. Start TensorBoard:
   ```bash
   tensorboard --logdir=runs/2026-02-10_001/logs/tensorboard
   ```

2. Navigate to the **DISTRIBUTIONS** or **HISTOGRAMS** tab

3. You'll see:
   - Time-series histograms showing how distributions change over training
   - Layer-by-layer parameter and gradient distributions
   - Color-coded visualization of distribution evolution

## Implementation Details

### Code Location

- **Config**: `configs/tensorboard.json`
- **Config Loader**: `utils_tensorboard/config.py` - `get_histogram_interval()`
- **Logger**: `utils_tensorboard/logger.py` - `log_model_histograms()`
- **Training Loop**: `train.py` - Called every `histogram_interval` steps

### Training Loop Integration

```python
# In train.py
histogram_interval = get_histogram_interval()

for iter in range(max_iters):
    # ... training code ...
    
    # Log histograms periodically
    if iter % histogram_interval == 0 or iter == max_iters - 1:
        tb_logger.log_model_histograms(model_instance, iter)
```

## Disabling Histogram Logging

To completely disable histogram logging and improve training performance:

1. Edit `configs/tensorboard.json`:
   ```json
   {
     "runs_directory": "runs",
     "default_port": 6006,
     "max_port_attempts": 10,
     "histogram_interval": 0
   }
   ```

2. Or use a negative value:
   ```json
   {
     "histogram_interval": -1
   }
   ```

When `histogram_interval <= 0`, the training loop will skip all histogram logging operations, eliminating any performance overhead.

## Performance Considerations

Histogram logging has some overhead:
- **CPU**: Moving tensors from GPU to CPU for logging
- **Disk**: Histogram data increases event file size
- **Time**: Additional I/O operations

**Recommendations**:
- Use `histogram_interval >= 100` for typical training runs
- Increase interval for very large models or long training runs
- **Set to 0 or negative to completely disable histogram logging**
- Disable by setting `histogram_interval: 0` if not needed

## Advanced Usage

### Custom Activation Logging

You can also log activation histograms using `log_activation_histograms()`:

```python
# Capture activations during forward pass
activations = {
    'layer1_output': layer1_output,
    'layer2_output': layer2_output,
}

# Log to TensorBoard
tb_logger.log_activation_histograms(activations, step=iter)
```

This is useful for:
- Monitoring activation distributions
- Detecting dead neurons (all zeros)
- Analyzing layer-wise information flow

## Troubleshooting

### Histograms not appearing in TensorBoard
- Check that `histogram_interval` is not too large
- Verify TensorBoard is reading from the correct log directory
- Refresh the TensorBoard page (F5)

### Large file sizes
- Increase `histogram_interval` to reduce logging frequency
- Consider logging only specific layers instead of all parameters

### Performance impact
- Monitor training speed with and without histogram logging
- Adjust `histogram_interval` to balance detail vs. performance
