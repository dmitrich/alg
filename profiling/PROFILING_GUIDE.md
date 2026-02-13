# Profiling Tools - Quick Reference Guide

## ✅ Installed Tools

1. **TensorBoard** - Visual profiling with PyTorch profiler
2. **torch-tb-profiler** - TensorBoard plugin for PyTorch
3. **memory-profiler** - Track memory usage
4. **line-profiler** - Line-by-line profiling
5. **snakeviz** - Interactive cProfile visualization
6. **py-spy** - Already installed (sampling profiler)

## 📁 Directory Structure

```
alg/
├── profiler_logs/          # TensorBoard profiling data
├── benchmark_results/      # Benchmark outputs
├── profiling_toolkit.py    # All-in-one profiling utilities
└── PROFILING_GUIDE.md      # This file
```

## 🚀 Quick Start

### 1. Lightweight Profiling (Console Output)

```python
from profiling_toolkit import ProfilingToolkit

@ProfilingToolkit.lightweight_profile
def training_step(model, x, y):
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    return loss

# Automatically profiles 10 iterations and prints results
loss = training_step(model, x, y)
```

### 2. TensorBoard Profiling (Visual)

```python
from profiling_toolkit import ProfilingToolkit

prof = ProfilingToolkit.tensorboard_profile()
prof.__enter__()

for step in range(max_iters):
    # Your training code
    prof.step()

prof.__exit__(None, None, None)
```

Then view in browser:
```bash
tensorboard --logdir=./profiler_logs
# Open: http://localhost:6006
```

### 3. Benchmark Specific Operations

```python
from profiling_toolkit import ProfilingToolkit

result = ProfilingToolkit.benchmark_operation(
    stmt='model(x)',
    setup='x = torch.randn(4, 64, device=device)',
    globals_dict={'model': model, 'device': device},
    num_runs=100
)
print(result)
```

### 4. Memory Profiling

```python
from profiling_toolkit import ProfilingToolkit

@ProfilingToolkit.memory_profile
def train_epoch():
    # Your training code
    pass
```

### 5. Simple Timing

```python
from profiling_toolkit import ProfilingToolkit

@ProfilingToolkit.time_function
def expensive_operation():
    # Your code
    pass
```

## 🔧 Command-Line Tools

### py-spy (Already used)
```bash
# Profile with GIL filter
sudo .venv/bin/py-spy record --gil -o profile.svg -- .venv/bin/python train.py

# Profile with thread info
sudo .venv/bin/py-spy record --threads -o profile.svg -- .venv/bin/python train.py
```

### cProfile + snakeviz
```bash
# Profile and visualize
python -m cProfile -o profile.prof train.py
snakeviz profile.prof
```

### line_profiler
```bash
# Add @profile decorator to functions
kernprof -l -v train.py
```

### memory_profiler
```bash
# Add @profile decorator to functions
python -m memory_profiler train.py
```

## 📊 What Each Tool Shows

| Tool | Best For | Output |
|------|----------|--------|
| **py-spy** | Overall CPU usage, thread activity | Flamegraph (SVG) |
| **PyTorch Profiler** | PyTorch operations, GPU/CPU breakdown | TensorBoard |
| **cProfile** | Python function calls | Stats table |
| **line_profiler** | Line-by-line bottlenecks | Per-line timing |
| **memory_profiler** | Memory usage per line | Memory graph |
| **torch.utils.benchmark** | Micro-benchmarks | Timing stats |

## 🎯 Optimization Workflow

1. **Start with py-spy** - Get overall picture
2. **Use PyTorch Profiler** - Identify slow operations
3. **Benchmark specific ops** - Measure improvements
4. **Profile memory** - Check for leaks
5. **Iterate** - Apply optimizations and re-profile

## 💡 Common Optimizations

### For M4/MPS:
- ✅ Increase batch size (better GPU utilization)
- ✅ Use larger models (amortize overhead)
- ✅ Reduce eval frequency
- ⚠️ torch.compile has minimal benefit on MPS
- ⚠️ Mixed precision not fully supported on MPS yet

### General:
- Use DataLoader with num_workers > 0
- Pin memory for faster GPU transfer
- Use gradient accumulation for larger effective batch size
- Profile and optimize data loading separately

## 📝 Next Steps

1. Run lightweight profiling on your training loop
2. Identify the slowest operations
3. Benchmark those operations individually
4. Apply targeted optimizations
5. Re-profile to measure improvement

---

**All tools are installed and ready to use!**
