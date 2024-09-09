# Adapted from https://github.com/cuda-mode/lectures/blob/main/lecture_001/pytorch_square.py

import torch
from torch.autograd import profiler

def create_test_tensor(size=(10000, 10000), device='cuda'):
    return torch.randn(size, device=device)

def square_methods():
    return {
        'torch.square': torch.square,
        'a * a': lambda a: a * a,
        'a ** 2': lambda a: a ** 2
    }

def compare_small_tensor():
    a = torch.tensor([1., 2., 3.])
    for name, func in square_methods().items():
        print(f"{name}: {func(a)}")

def cuda_warmup(input_tensor):
    # Perform a series of operations to ensure CUDA is fully initialized
    for _ in range(10):
        torch.cuda.synchronize()
        _ = torch.square(input_tensor)
        torch.cuda.synchronize()

def time_pytorch_function(func, input_tensor, num_runs=100):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warm-up run
    for _ in range(10):
        func(input_tensor)
    
    # Actual timing
    torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(num_runs):
        func(input_tensor)
    
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / num_runs

def benchmark_methods(input_tensor):
    cuda_warmup(input_tensor)  # Ensure CUDA is fully initialized
    results = {}
    for name, func in square_methods().items():
        elapsed_time = time_pytorch_function(func, input_tensor)
        results[name] = elapsed_time
    return results

def profile_method(func, input_tensor):
    cuda_warmup(input_tensor)  # Ensure CUDA is fully initialized
    with profiler.profile(use_cuda=True) as prof:
        func(input_tensor)
    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

def run_profiling(input_tensor):
    for name, func in square_methods().items():
        print(f"============")
        print(f"Profiling {name}")
        print(f"============")
        print(profile_method(func, input_tensor))

def main():
    # Compare small tensor results
    compare_small_tensor()
    
    # Create large tensor for benchmarking and profiling
    large_tensor = create_test_tensor()
    
    # Benchmark methods
    benchmark_results = benchmark_methods(large_tensor)
    for name, time in benchmark_results.items():
        print(f"{name}: {time:.4f} ms")
    
    # Run profiling
    run_profiling(large_tensor)

if __name__ == "__main__":
    main()