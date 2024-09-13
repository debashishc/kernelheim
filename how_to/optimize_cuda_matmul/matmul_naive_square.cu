#include <cuda_runtime.h> // Include this to use CUDA runtime functions
#include <stdio.h>


__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {

    #define N 512  // Define the size of matrices 

    // Allocate host memory
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize host matrices with arbitrary values (for testing)
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;  // Example: Fill with ones for simplicity
        h_B[i] = 1.0f;  // Example: Fill with ones for simplicity
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);  // Make sure to cast with (void**)
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block size and grid size
    int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the matrix multiplication kernel
    matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish before accessing results
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Validate the result (simple validation for testing)
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != N) {
            printf("Error: Element C[%d] = %f, expected %d\n", i, h_C[i], N);
            break;
        }
    }

    printf("Matrix multiplication completed successfully.\n");

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
