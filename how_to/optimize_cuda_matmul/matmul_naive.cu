// Matrix multiplication kernel: C = A * B
// Each thread computes one element of the output matrix C

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void MatrixMulKernel(float* A, float* B, float* C, int M, int N, int P)
{
    // Calculate the row index of the C element to work on
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the column index of the C element to work on
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to ensure we don't access beyond the matrices
    if (row < M && col < P)
    {
        float value = 0;

        // Compute the dot product of the row of A and column of B
        for (int k = 0; k < N; ++k)
        {
            value += A[row * N + k] * B[k * P + col];
        }

        // Write the computed value to the output matrix
        C[row * P + col] = value;
    }
}



int main()
{
    // Define matrix dimensions
    int M = 1024; // Number of rows in A and C
    int N = 1024; // Number of columns in A and rows in B
    int P = 1024; // Number of columns in B and C

    // Allocate host memory for matrices A, B, and C
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * P * sizeof(float);
    size_t size_C = M * P * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // Initialize host matrices h_A and h_B (fill with data)
    // For simplicity, we can initialize them with some values
    for (int i = 0; i < M * N; ++i)
        h_A[i] = 1.0f; // or any other value

    for (int i = 0; i < N * P; ++i)
        h_B[i] = 1.0f; // or any other value

    // Allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // Copy host matrices A and B to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Set up block and grid dimensions
    dim3 blockSize(16, 16);  // Number of threads per block in x and y
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    // The grid dimensions ensure that we cover all elements of the output matrix C

    // Launch the matrix multiplication kernel
    MatrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, P);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Use the result matrix h_C (e.g., print or validate the results)
    // For example, we can print the first element
    printf("C[0] = %f\n", h_C[0]);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
