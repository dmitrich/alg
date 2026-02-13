"""
Benchmark script for GPT model operations
"""
import torch
import torch.utils.benchmark as benchmark

# Import and set module-level config
import model
model.vocab_size = 65
model.n_embd = 64
model.n_head = 4
model.n_layer = 4
model.block_size = 64
model.dropout = 0.0

from model import BigramLanguageModel

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Benchmarking on device: {device}\n")

# Create model
m = BigramLanguageModel().to(device)
m.eval()

# Test data
batch_size = 4
seq_len = 64
x = torch.randint(0, 65, (batch_size, seq_len), device=device)
y = torch.randint(0, 65, (batch_size, seq_len), device=device)

print("="*80)
print("BENCHMARK RESULTS")
print("="*80)

# Benchmark 1: Forward pass
print("\n1. FORWARD PASS")
print("-"*80)
t = benchmark.Timer(stmt='m(x)', globals={'m': m, 'x': x})
print(f"   {t.timeit(100)}")

# Benchmark 2: Forward + Loss
print("\n2. FORWARD + LOSS")
print("-"*80)
t = benchmark.Timer(stmt='m(x, y)', globals={'m': m, 'x': x, 'y': y})
print(f"   {t.timeit(100)}")

# Benchmark 3: Single block
print("\n3. SINGLE TRANSFORMER BLOCK")
print("-"*80)
block = m.blocks[0]
hidden = torch.randn(batch_size, seq_len, 64, device=device)
t = benchmark.Timer(stmt='block(hidden)', globals={'block': block, 'hidden': hidden})
print(f"   {t.timeit(100)}")

# Benchmark 4: Attention
print("\n4. MULTI-HEAD ATTENTION")
print("-"*80)
attention = m.blocks[0].sa
t = benchmark.Timer(stmt='attention(hidden)', globals={'attention': attention, 'hidden': hidden})
print(f"   {t.timeit(100)}")

# Benchmark 5: FFN
print("\n5. FEED-FORWARD NETWORK")
print("-"*80)
ffn = m.blocks[0].ffwd
t = benchmark.Timer(stmt='ffn(hidden)', globals={'ffn': ffn, 'hidden': hidden})
print(f"   {t.timeit(100)}")

# Benchmark 6: Embedding
print("\n6. TOKEN EMBEDDING")
print("-"*80)
t = benchmark.Timer(stmt='m.token_embedding_table(x)', globals={'m': m, 'x': x})
print(f"   {t.timeit(100)}")

print("\n" + "="*80)
print("✅ BENCHMARK COMPLETE")
print("="*80)
