
"""
Train with TensorBoard Profiler
Run this to get detailed visual profiling in TensorBoard
"""
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Setup profiler
prof = profile(
    activities=[ProfilerActivity.CPU],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)

print("\n" + "="*80)
print("TENSORBOARD PROFILING ENABLED")
print("="*80)
print("Profiling schedule:")
print("  - Wait: 1 step")
print("  - Warmup: 1 step")
print("  - Active profiling: 3 steps")
print("  - Repeat: 2 times")
print("  - Total profiled steps: 10")
print("="*80 + "\n")

# Import training code
import sys
sys.path.insert(0, '.')

# Load model and setup
exec(open('train.py').read().split('# Training loop')[0])

# Training loop with profiler
prof.start()
for iter_num in range(20):  # Profile 20 iterations
    xb, yb = get_batch('train', train_data, val_data, config_obj.block_size, config_obj.batch_size, config_obj.device)
    
    logits, loss = model_instance(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    prof.step()
    
    if iter_num % 5 == 0:
        print(f"Profiled iteration {iter_num+1}/20")

prof.stop()

print("\n" + "="*80)
print("PROFILING COMPLETE!")
print("="*80)
print("\nTo view results:")
print("  1. Run: tensorboard --logdir=./profiler_logs")
print("  2. Open browser: http://localhost:6006")
print("  3. Click on 'PYTORCH_PROFILER' tab")
print("="*80)
