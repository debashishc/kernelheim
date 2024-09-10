# Profiling CUDA

[torch.profiler](https://pytorch.org/docs/stable/profiler.html)


Profiling torch.square
============
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
             aten::square         0.75%      12.979us         7.60%     130.642us     130.642us      15.000us         0.85%       1.763ms       1.763ms             1  
                aten::pow         4.77%      81.962us         6.52%     112.112us     112.112us       1.737ms        98.53%       1.748ms       1.748ms             1  
        aten::result_type         0.10%       1.760us         0.10%       1.760us       1.760us       6.000us         0.34%       6.000us       6.000us             1  
                 aten::to         0.06%       0.950us         0.06%       0.950us       0.950us       5.000us         0.28%       5.000us       5.000us             1  
          cudaEventRecord         1.42%      24.461us         1.42%      24.461us       3.058us       0.000us         0.00%       0.000us       0.000us             8  
         cudaLaunchKernel         1.08%      18.570us         1.08%      18.570us      18.570us       0.000us         0.00%       0.000us       0.000us             1  
    cudaDeviceSynchronize        91.82%       1.579ms        91.82%       1.579ms       1.579ms       0.000us         0.00%       0.000us       0.000us             1  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.720ms
Self CUDA time total: 1.763ms

Description of Profiler Output:


| **Column Name**        | **Definition**                                                                                                                                 |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Name**               | The operation or function being profiled (e.g., `aten::pow`, `cudaLaunchKernel`). `aten::` refers to PyTorch’s internal operations, while `cuda::` refers to GPU-specific operations. |
| **Self CPU %**         | The percentage of the total CPU time spent only in this specific function (excluding time spent in functions it calls).                       |
| **Self CPU**           | The actual amount of time (in microseconds or milliseconds) spent on the CPU in this function itself, excluding time spent in any nested functions. |
| **CPU total %**        | The percentage of total CPU time spent in this function and any functions it calls (inclusive).                                                |
| **CPU total**          | The actual total time (in microseconds or milliseconds) spent on the CPU in this function, including any functions it calls.                  |
| **CPU time avg**       | The average time spent on the CPU per call of the function.                                                                                    |
| **Self CUDA**          | The actual time (in microseconds or milliseconds) spent on the GPU (CUDA) for this function itself, excluding any nested calls.               |
| **Self CUDA %**        | The percentage of total GPU (CUDA) time spent in this function itself (excluding nested calls).                                                |
| **CUDA total**         | The total time (in microseconds or milliseconds) spent on the GPU in this function, including any nested functions it calls.                  |
| **CUDA time avg**      | The average time spent on the GPU (CUDA) per call of the function.                                                                             |
| **# of Calls**         | The number of times the function or operation was called during the profiling session.                                                         |

