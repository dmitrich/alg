"""
Profiling Toolkit - All-in-one profiling utilities
"""
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import time
from functools import wraps

class ProfilingToolkit:
    """Comprehensive profiling utilities for PyTorch training"""
    
    @staticmethod
    def lightweight_profile(func, num_iterations=10):
        """Quick profiling of a function - prints to console"""
        def wrapper(*args, **kwargs):
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("profiled_function"):
                    for _ in range(num_iterations):
                        result = func(*args, **kwargs)
            
            print("\n" + "="*80)
            print(f"PROFILE RESULTS (averaged over {num_iterations} iterations)")
            print("="*80)
            print(prof.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=20
            ))
            return result
        return wrapper
    
    @staticmethod
    def tensorboard_profile(log_dir='./profiler_logs', wait=1, warmup=1, active=3, repeat=2):
        """Full TensorBoard profiling - use with training loop"""
        return profile(
            activities=[ProfilerActivity.CPU],
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
    
    @staticmethod
    def benchmark_operation(stmt, setup='', globals_dict=None, num_runs=100):
        """Benchmark a specific operation"""
        import torch.utils.benchmark as benchmark
        
        timer = benchmark.Timer(
            stmt=stmt,
            setup=setup,
            globals=globals_dict or {}
        )
        
        result = timer.timeit(num_runs)
        return result
    
    @staticmethod
    def memory_profile(func):
        """Profile memory usage"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.backends.mps.is_available():
                # MPS doesn't have memory stats yet
                print("Memory profiling not available for MPS")
                return func(*args, **kwargs)
            
            torch.cuda.reset_peak_memory_stats()
            result = func(*args, **kwargs)
            
            if torch.cuda.is_available():
                print(f"\nMemory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            
            return result
        return wrapper
    
    @staticmethod
    def time_function(func):
        """Simple timing decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"\n{func.__name__} took {elapsed:.4f} seconds")
            return result
        return wrapper


# Example usage functions
def example_lightweight_profiling():
    """Example: Quick profiling"""
    print("""
# Add to your code:
from profiling_toolkit import ProfilingToolkit

@ProfilingToolkit.lightweight_profile
def training_step(model, x, y):
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    return loss

# Run once - it will profile 10 iterations
loss = training_step(model, x, y)
""")

def example_tensorboard_profiling():
    """Example: TensorBoard profiling"""
    print("""
# Add to your training loop:
from profiling_toolkit import ProfilingToolkit

prof = ProfilingToolkit.tensorboard_profile()
prof.__enter__()

for step in range(max_iters):
    # Your training code
    train_step()
    prof.step()

prof.__exit__(None, None, None)

# Then run: tensorboard --logdir=./profiler_logs
# Open browser: http://localhost:6006
""")

def example_benchmarking():
    """Example: Benchmark specific operations"""
    print("""
# Benchmark forward pass:
from profiling_toolkit import ProfilingToolkit

result = ProfilingToolkit.benchmark_operation(
    stmt='model(x)',
    setup='x = torch.randn(4, 64, device=device)',
    globals_dict={'model': model, 'device': device},
    num_runs=100
)
print(f"Forward pass: {result}")
""")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PROFILING TOOLKIT - USAGE EXAMPLES")
    print("="*80)
    
    print("\n1. LIGHTWEIGHT PROFILING (Quick console output)")
    print("-"*80)
    example_lightweight_profiling()
    
    print("\n2. TENSORBOARD PROFILING (Visual analysis)")
    print("-"*80)
    example_tensorboard_profiling()
    
    print("\n3. BENCHMARKING (Micro-benchmarks)")
    print("-"*80)
    example_benchmarking()
    
    print("\n" + "="*80)
    print("TOOLKIT READY TO USE!")
    print("="*80)
    print("\nImport in your code:")
    print("  from profiling_toolkit import ProfilingToolkit")
