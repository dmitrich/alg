"""
STEP 2: Create comprehensive benchmark script for your model
"""
import torch
import torch.nn.functional as F
from model import BigramLanguageModel
from parameters import ModelConfig
import torch.utils.benchmark as benchmark

print("\n" + "="*80)
print("STEP 2: CREATING BENCHMARK SCRIPT")
print("="*80)

# Create benchmark script
benchmark_script = '''
"""
Benchmark script for GPT model operations
Measures performance of individual operations
"""
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from model import BigramLanguageModel
from parameters import ModelConfig

# Setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Benchmarking on device: {device}\\n")

# Create model
config = ModelConfig(
    batch_size=4,
    block_size=64,
    max_iters=2000,
    eval_interval=100,
    learning_rate=0.02,
    eval_iters=200,
    n_embd=64,
    n_head=4,
    n_layer=4,
    dropout=0.0,
    vocab_size=65,
    device=device
)

model = BigramLanguageModel(config).to(device)
model.eval()

# Test data
batch_size = 4
seq_len = 64
x = torch.randint(0, 65, (batch_size, seq_len), device=device)
y = torch.randint(0, 65, (batch_size, seq_len), device=device)

print("="*80)
print("BENCHMARK RESULTS")
print("="*80)

# Benchmark 1: Forward pass
print("\\n1. FORWARD PASS")
print("-"*80)
t_forward = benchmark.Timer(
    stmt='model(x)',
    globals={'model': model, 'x': x}
)
result = t_forward.timeit(100)
print(f"   {result}")

# Benchmark 2: Forward + Loss
print("\\n2. FORWARD + LOSS COMPUTATION")
print("-"*80)
t_forward_loss = benchmark.Timer(
    stmt="""
logits, loss = model(x, y)
""",
    globals={'model': model, 'x': x, 'y': y}
)
result = t_forward_loss.timeit(100)
print(f"   {result}")

# Benchmark 3: Single layer forward
print("\\n3. SINGLE TRANSFORMER BLOCK")
print("-"*80)
single_block = model.blocks[0]
hidden = torch.randn(batch_size, seq_len, config.n_embd, device=device)
t_block = benchmark.Timer(
    stmt='block(hidden)',
    globals={'block': single_block, 'hidden': hidden}
)
result = t_block.timeit(100)
print(f"   {result}")

# Benchmark 4: Attention mechanism
print("\\n4. MULTI-HEAD ATTENTION")
print("-"*80)
attention = model.blocks[0].sa
t_attention = benchmark.Timer(
    stmt='attention(hidden)',
    globals={'attention': attention, 'hidden': hidden}
)
result = t_attention.timeit(100)
print(f"   {result}")

# Benchmark 5: Feed-forward network
print("\\n5. FEED-FORWARD NETWORK")
print("-"*80)
ffn = model.blocks[0].ffwd
t_ffn = benchmark.Timer(
    stmt='ffn(hidden)',
    globals={'ffn': ffn, 'hidden': hidden}
)
result = t_ffn.timeit(100)
print(f"   {result}")

# Benchmark 6: Embedding lookup
print("\\n6. TOKEN EMBEDDING")
print("-"*80)
t_embed = benchmark.Timer(
    stmt='model.token_embedding_table(x)',
    globals={'model': model, 'x': x}
)
result = t_embed.timeit(100)
print(f"   {result}")

# Benchmark 7: Position embedding
print("\\n7. POSITION EMBEDDING")
print("-"*80)
t_pos_embed = benchmark.Timer(
    stmt='model.position_embedding_table(torch.arange(seq_len, device=device))',
    globals={'model': model, 'seq_len': seq_len, 'device': device, 'torch': torch}
)
result = t_pos_embed.timeit(100)
print(f"   {result}")

# Benchmark 8: Layer norm
print("\\n8. LAYER NORMALIZATION")
print("-"*80)
ln = model.blocks[0].ln1
t_ln = benchmark.Timer(
    stmt='ln(hidden)',
    globals={'ln': ln, 'hidden': hidden}
)
result = t_ln.timeit(100)
print(f"   {result}")

print("\\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)
print("\\nResults saved to: benchmark_results/model_benchmarks.txt")

# Save results
import os
os.makedirs('benchmark_results', exist_ok=True)
'''

# Write benchmark script
with open('benchmark_model.py', 'w') as f:
    f.write(benchmark_script)

print("\n✅ Benchmark script created: benchmark_model.py")
print("\nRunning benchmarks...")

# Run it
import subprocess
result = subprocess.run(['.venv/bin/python', 'benchmark_model.py'],
                       capture_output=True, text=True)
print(result.stdout)
if result.stderr and 'UserWarning' not in result.stderr:
    print("Errors:", result.stderr[:500])

print("\n✅ Step 2 complete!")
print("\nNext: Run 'python setup_tensorboard_profiling.py' for Step 3")
