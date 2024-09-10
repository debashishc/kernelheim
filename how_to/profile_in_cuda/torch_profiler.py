# Adapted from https://github.com/cuda-mode/lectures/blob/main/lecture_001/pt_profiler.py

import torch
from torch.profiler import profile, ProfilerActivity

# Custom trace handler that prints a performance summary and exports a trace file
def trace_handler(profiler):
    # Print a summary of the profiling sorted by total CUDA self-time (GPU time)
    print(profiler.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    
    # Export the profiler trace as a Chrome-compatible JSON file
    trace_file = f"./profiler/test_trace_{profiler.step_num}.json"
    profiler.export_chrome_trace(trace_file)
    print(f"Trace saved to {trace_file}")

# Main function to run profiling
def run_profiling():
    # Define the profiling activities (CPU and GPU)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # Set up a custom profiler schedule
    schedule = torch.profiler.schedule(
        wait=1,    # Skip the first iteration
        warmup=1,  # Warm-up during the second iteration
        active=2,  # Actively profile the third and fourth iterations
        repeat=1   # Repeat the cycle after profiling the active iterations
    )

    # Use the profiler with a custom schedule and trace handler
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler   # Call trace_handler when a trace is ready
    ) as profiler:
        
        # Perform tensor operations across multiple iterations
        for iteration in range(10):
            # Perform a tensor square operation on the GPU
            result = torch.square(torch.randn(10000, 10000).cuda())
            
            # Signal the profiler that this iteration has finished
            profiler.step()

if __name__ == "__main__":
    run_profiling()
