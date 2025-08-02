#include <iostream>
#include <cuda_runtime.h>
#include <chrono>  // Include for timing
#define N 1028
#define BLOCK_SIZE 16
#define TILE_SIZE 32

__global__ void matrixMulKernel(float *A, float *B, float *C, int width) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (width + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load data into shared memory
        if (row < width && (tile * TILE_SIZE + tx) < width)
            sA[ty][tx] = A[row * width + tile * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;
            
        if (col < width && (tile * TILE_SIZE + ty) < width)
            sB[ty][tx] = B[(tile * TILE_SIZE + ty) * width + col];
        else
            sB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum for the current tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
};

int main() {
    float *h_A, *h_B, *h_C; // Host matrices
    float *d_A, *d_B, *d_C; // Device matrices

    // Allocate memory on host
    h_A = (float*)malloc(N * N * sizeof(float));
    h_B = (float*)malloc(N * N * sizeof(float));
    h_C = (float*)malloc(N * N * sizeof(float));

    // Initialize host matrices A and B with more cache-friendly pattern
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = rand() % 10;
            h_B[i * N + j] = rand() % 10;
        }
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Warm up the GPU
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    //Ensure that the kernel has finished
    cudaDeviceSynchronize();

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result (optional)
    // You can add code here to compare h_C with the expected result

    // Free memory on device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free memory on host
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Matrix multiplication complete!" << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
// Compile with: nvcc -o pinak pinak.cu
// Run with: ./pinak