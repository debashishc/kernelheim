#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 1024  // Define the size of each block

// CUDA Kernel to compute the exponentials of logits
__global__ void exponentiate(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);  // Exponentiate each value
    }
}

// CUDA Kernel to sum the exponentials using parallel reduction
__global__ void sum_exp(float *input, float *output, int size) {
    __shared__ float shared_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write the block's partial sum to the output array
    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

// CUDA Kernel to normalize the logits by dividing each exponentiated value by the sum
__global__ void normalize(float *input, float *sum_exp, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / (*sum_exp);
    }
}

// Main function to launch the kernels
void softmax(float *logits, float *softmax_output, int size) {
    float *d_logits, *d_exp_logits, *d_softmax_output, *d_sum_exp;
    float h_sum_exp = 0.0f;

    // Allocate memory on the device
    cudaMalloc((void **)&d_logits, size * sizeof(float));
    cudaMalloc((void **)&d_exp_logits, size * sizeof(float));
    cudaMalloc((void **)&d_softmax_output, size * sizeof(float));
    cudaMalloc((void **)&d_sum_exp, sizeof(float));

    // Copy logits to device
    cudaMemcpy(d_logits, logits, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to exponentiate the logits
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exponentiate<<<num_blocks, BLOCK_SIZE>>>(d_logits, d_exp_logits, size);

    // Initialize sum_exp on device to 0
    cudaMemcpy(d_sum_exp, &h_sum_exp, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to sum the exponentiated values using parallel reduction
    sum_exp<<<num_blocks, BLOCK_SIZE>>>(d_exp_logits, d_sum_exp, size);

    // Copy the sum of exponentiated values back to the host
    cudaMemcpy(&h_sum_exp, d_sum_exp, sizeof(float), cudaMemcpyDeviceToHost);

    // Launch kernel to normalize the exponentiated logits
    normalize<<<num_blocks, BLOCK_SIZE>>>(d_exp_logits, d_sum_exp, d_softmax_output, size);

    // Copy the softmax output back to host
    cudaMemcpy(softmax_output, d_softmax_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_logits);
    cudaFree(d_exp_logits);
    cudaFree(d_softmax_output);
    cudaFree(d_sum_exp);
}

// Host-side code to test the CUDA softmax
int main() {
    const int size = 10;
    float logits[size] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float softmax_output[size];

    // Run softmax on the logits
    softmax(logits, softmax_output, size);

    // Print the output
    printf("Softmax Output:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", softmax_output[i]);
    }
    printf("\n");

    return 0;
}
