# Naive Matrix Multiplication

In Native Matrix Multiplication, the ratio of FLOPs to bytes accessed from the global memory is 2 FLOPS to 8 bytes. The **2 FLOP to 8 bytes** ratio comes from the fact that for each element of the resulting matrix, you perform 2 floating-point operations (1 multiplication and 1 addition) and access 2 elements from global memory (one from matrix \(A\) and one from matrix \(B\)), each element being 4 bytes. This leads to a relatively low arithmetic intensity, indicating that the performance is constrained by global memory access rather than computation i.e. memory bound rather than compute bound.

```sh {"id":"01J7QM2JMJR87Y1QETK145P8Y1"}
for (int i = 0, i < width; ++i ) {
    sum += A[row * width + i] + B[i * width + col]
}
```


Arithmetic Intensity/ Computational Intensity: The **compute to global memory access ratio** (often referred to as the **arithmetic intensity**) is a key metric that describes the ratio of the number of floating-point operations (FLOP) to the amount of memory being accessed.

### Understanding Naive Matrix Multiplication

For matrix multiplication, the formula is:

\[
C[i, j] = \sum_{k=0}^{N-1} A[i, k] \times B[k, j]
\]

Where:
- \(A\) is an \(N \times N\) matrix,
- \(B\) is an \(N \times N\) matrix,
- \(C\) is the resulting \(N \times N\) matrix.

In this naive GPU kernel, each thread computes one element \(C[i, j]\) of the output matrix.

### Breakdown of the Compute and Memory Access

#### 1. **Compute (FLOP)**

For each element \(C[i, j]\), the computation involves:
- One **multiplication** operation per iteration of the inner loop (for each \(k\)).
- One **addition** operation per iteration of the inner loop (to accumulate the sum for \(C[i, j]\)).

So for each \(k\), there are 2 FLOP:
- 1 FLOP for the multiplication \(A[i, k] \times B[k, j]\),
- 1 FLOP for the addition to the accumulator.

Hence, the **compute cost** is 2 FLOP per iteration of the inner loop (for each \(k\)).

#### 2. **Global Memory Access**

For each element \(C[i, j]\) that a thread computes:
- You need to **read one element of \(A[i, k]\)** (from global memory),
- You need to **read one element of \(B[k, j]\)** (from global memory),
- Optionally, you may also read and write to \(C[i, j]\) if it's not cached.

In a naive implementation:
- **2 global memory reads** occur for each iteration of the inner loop:
  - 1 read from matrix \(A\) for \(A[i, k]\),
  - 1 read from matrix \(B\) for \(B[k, j]\).

These memory accesses involve fetching data from global memory, which typically stores **single-precision floating-point** values as **4 bytes** each. Therefore, you read **8 bytes** per iteration (4 bytes from \(A[i, k]\) and 4 bytes from \(B[k, j]\)).

So, for each iteration, the **memory access cost** is 8 bytes (2 reads of 4 bytes each).

### Compute to Memory Access Ratio

Now let's calculate the compute to global memory access ratio for each iteration:

- **2 FLOP** (1 multiplication and 1 addition),
- **8 bytes** (2 global memory reads, 4 bytes each).

Thus, the **compute to memory access ratio** is:

\[
\text{Compute to memory access ratio} = \frac{\text{2 FLOP}}{\text{8 bytes}} = 0.25 \text{ FLOP per byte}
\]

This ratio means that for every byte of data transferred from global memory, only 0.25 floating-point operations are performed, which is quite low. This low ratio indicates that the naive implementation is **memory-bound**, meaning the performance is primarily limited by memory bandwidth rather than the compute capacity of the GPU.

### Conclusion



In more optimized implementations (e.g., using shared memory, tiling, or caching), you can significantly reduce the number of global memory accesses per compute operation, leading to a higher compute to memory access ratio and better performance.