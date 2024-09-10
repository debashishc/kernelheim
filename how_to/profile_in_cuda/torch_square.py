# Adapted from https://github.com/cuda-mode/lectures/blob/main/lecture_001/pytorch_square.py
# Nice blog post from CJ Mills here: https://christianjmills.com/posts/cuda-mode-notes/lecture-001/ 

import torch
from torch.autograd import profiler
from typing import Callable, Dict, Tuple

def create_large_tensor(size: Tuple[int, int] = (10000, 10000), device: str = 'cuda') -> torch.Tensor:
    """
    Creates a large tensor of random numbers on the specified device.
    
    Args:
        size (Tuple[int, int]): Size of the tensor (default is (10000, 10000)).
        device (str): Device where the tensor will be created, e.g., 'cuda' or 'cpu' (default is 'cuda').
    
    Returns:
        torch.Tensor: A tensor filled with random numbers.
    """
    return torch.randn(size, device=device)

def square_methods() -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Returns a dictionary of methods for squaring a tensor, including:
    - `torch.square`: Direct usage of PyTorch's square function.
    - `a * a`: Manual squaring using multiplication.
    - `a ** 2`: Manual squaring using exponentiation.
    
    Returns:
        Dict[str, Callable[[torch.Tensor], torch.Tensor]]: A dictionary of methods for squaring a tensor.
    """
    return {
        'torch.square': torch.square,
        'a * a': lambda a: a * a,
        'a ** 2': lambda a: a ** 2
    }

def compare_small_tensor() -> None:
    """
    Compares different square methods on a small tensor and prints the results.
    Used for correctness testing.
    
    Returns:
        None
    """
    a = torch.tensor([1., 2., 3.])
    for name, func in square_methods().items():
        print(f"{name}: {func(a)}")

def cuda_warmup(input_tensor: torch.Tensor) -> None:
    """
    Ensures CUDA is fully initialized by performing a series of operations.
    
    Args:
        input_tensor (torch.Tensor): Tensor used for warming up CUDA operations.
    
    Returns:
        None
    """
    torch.cuda.synchronize()
    for _ in range(10):
        _ = torch.square(input_tensor)
    torch.cuda.synchronize()

def time_pytorch_function(func: Callable[[torch.Tensor], torch.Tensor], 
                          input_tensor: torch.Tensor, 
                          num_runs: int = 100) -> float:
    """
    Times a PyTorch function using CUDA events and returns the average execution time.
    
    Args:
        func (Callable[[torch.Tensor], torch.Tensor]): The function to time.
        input_tensor (torch.Tensor): The tensor input for the function.
        num_runs (int): Number of runs for averaging (default is 100).
    
    Returns:
        float: Average execution time in milliseconds.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warm-up run
    for _ in range(10):
        func(input_tensor)
    
    # Synchronize and start timing
    torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(num_runs):
        func(input_tensor)
    
    end_event.record()
    torch.cuda.synchronize()
    
    # Return the average time per run
    return start_event.elapsed_time(end_event) / num_runs

def benchmark_methods(input_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Benchmarks the square methods using the input tensor and returns the results.
    
    Args:
        input_tensor (torch.Tensor): Tensor used for benchmarking the square methods.
    
    Returns:
        Dict[str, float]: A dictionary containing the method names and their corresponding average execution times in milliseconds.
    """
    cuda_warmup(input_tensor)
    results = {}
    for name, func in square_methods().items():
        elapsed_time = time_pytorch_function(func, input_tensor)
        results[name] = elapsed_time
    return results

def profile_method(func: Callable[[torch.Tensor], torch.Tensor], input_tensor: torch.Tensor) -> str:
    """
    Profiles a specific method using PyTorch's autograd profiler and returns a summary.
    
    Args:
        func (Callable[[torch.Tensor], torch.Tensor]): The function to profile.
        input_tensor (torch.Tensor): The tensor input for the function.
    
    Returns:
        str: A string summary of the profiling results, sorted by CUDA time.
    """
    cuda_warmup(input_tensor)
    with profiler.profile(use_cuda=True) as prof:
        func(input_tensor)
    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

def run_profiling(input_tensor: torch.Tensor) -> None:
    """
    Profiles each square method and prints the profiling results for each.
    
    Args:
        input_tensor (torch.Tensor): Tensor used for profiling the square methods.
    
    Returns:
        None
    """
    for name, func in square_methods().items():
        print(f"============")
        print(f"Profiling {name}")
        print(f"============")
        print(profile_method(func, input_tensor))

def main() -> None:
    """
    Main function that performs the following steps:
    1. Compares small tensor results to check correctness of square methods.
    2. Creates a large tensor for benchmarking and profiling.
    3. Benchmarks the square methods and prints the results.
    4. Profiles the square methods and prints the profiling results.
    
    Returns:
        None
    """
    # Compare correctness of small tensor results
    compare_small_tensor()
    
    # Create large tensor for benchmarking and profiling
    large_tensor = create_large_tensor()
    
    # Benchmark methods
    print("Benchmarking methods:")
    benchmark_results = benchmark_methods(large_tensor)
    for name, time in benchmark_results.items():
        print(f"{name}: {time:.4f} ms")
    
    # Run profiling on the large tensor
    print("\nRunning profiling on large tensor methods:")
    run_profiling(large_tensor)

if __name__ == "__main__":
    main()

