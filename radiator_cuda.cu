#include <stdio.h>
#include <string.h>
#include <time.h>

#include "cuda_debug.h"
#include "radiator_cuda.h"

#define BLOCK_SIZE_X 256
#define BLOCK_SIZE_Y 1
#define BLOCK_SIZE_X_AVG 64
#define BLOCK_SIZE_Y_AVG 1
#define MAX_SHARED_MEM 32768
#define SHARED_MEM

// Device-side function to compute one cell update
__device__ __forceinline__ FLOAT row_iteration_cuda(FLOAT *matrix_old, int i, int j, int m)
{
  FLOAT res = 1.6f * matrix_old[i * m + (j - 2 + m) % m] + 
              1.55f * matrix_old[i * m + (j - 1 + m) % m] + 
              matrix_old[i * m + j % m] + 
              0.6f * matrix_old[i * m + (j + 1) % m] + 
              0.25f * matrix_old[i * m + (j + 2) % m];
  return res * 0.2f;
}

#ifndef SHARED_MEM
// Global-memory kernel
__global__ void propagate_row_heat_per_block(FLOAT *matrix_a, FLOAT *matrix_b,int n, int m, int iterations)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n) return;
  
  int base_idx = i * m;
  int blocks = (m + blockDim.x - 1) / blockDim.x;
  
  for (int it = 0; it < iterations; ++it)
  {
    // Only process interior cells (skip column 0)
    for (int b = 0; b < blocks; ++b)
    {
      int j = b * blockDim.x + threadIdx.x;
      if (j > 0 && j < m - 1)
      {
        matrix_b[base_idx + j] = row_iteration_cuda(matrix_a, i, j, m);
      }
    }
    __syncthreads();
    
    // Swap pointers
    FLOAT *tmp = matrix_a;
    matrix_a = matrix_b;
    matrix_b = tmp;
  }
}
#else
// Simplified shared memory version with explicit column 0 preservation
__global__ void propagate_row_heat_per_block_with_shared(FLOAT *matrix_a,FLOAT *matrix_b,int n, int m,int iterations)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n) return;
  
  int idx = threadIdx.x;
  int base_idx = i * m;
  int blocks = (m + blockDim.x - 1) / blockDim.x;
  
  for (int it = 0; it < iterations; ++it)
  {
    // Only process interior cells (skip column 0)
    for (int b = 0; b < blocks; ++b)
    {
      int j = b * blockDim.x + idx;
      if (j > 0 && j < m - 1)
      {
        matrix_b[base_idx + j] = row_iteration_cuda(matrix_a, i, j, m);
      }
    }
    __syncthreads();
    
    // Swap pointers
    FLOAT *tmp = matrix_a;
    matrix_a = matrix_b;
    matrix_b = tmp;
  }
}
#endif

__global__ void average_row_heat(FLOAT *matrix, int n, int m, FLOAT *out)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= n) return;
  
  extern __shared__ FLOAT row_sum[];
  row_sum[threadIdx.y] = 0.0f;
  __syncthreads();
  
  for (int x = threadIdx.x + 1; x < m - 1; x += blockDim.x)
  {
    atomicAdd(&row_sum[threadIdx.y], matrix[y * m + x]);
  }
  __syncthreads();
  
  if (threadIdx.x == 0)
    out[y] = row_sum[threadIdx.y] / (m - 2);
}

// C linkage for host code
extern "C" void cuda_propagate_heat(FLOAT *matrix, int n,int m,int iterations,float *timings,FLOAT *averages, uint8_t average)
{

  if (m % BLOCK_SIZE_X != 0 || n % BLOCK_SIZE_Y != 0) {
    fprintf(stderr, "Error: block size must divide matrix dimensions\n");
    exit(EXIT_FAILURE);
  }
  
  FLOAT *devA = nullptr, *devB = nullptr, *devAvg = nullptr;
  cudaEvent_t start, finish;
  cudaEventCreate(&start);
  cudaEventCreate(&finish);

  // alloc
  cudaEventRecord(start, 0);
  CUDA_CHECK(cudaMalloc(&devA, n * m * sizeof(FLOAT)), DEBUG_MALLOC);
  CUDA_CHECK(cudaMalloc(&devB, n * m * sizeof(FLOAT)), DEBUG_MALLOC);
  if (average)
    CUDA_CHECK(cudaMalloc(&devAvg, n * sizeof(FLOAT)), DEBUG_MALLOC);
  cudaEventRecord(finish, 0);
  cudaEventSynchronize(finish);
  cudaEventElapsedTime(&timings[0], start, finish);

  // copy in
  cudaEventRecord(start, 0);
  CUDA_CHECK(cudaMemcpy(devA, matrix, n * m * sizeof(FLOAT), cudaMemcpyHostToDevice), DEBUG_MEMCPY);
  CUDA_CHECK(cudaMemcpy(devB, matrix, n * m * sizeof(FLOAT), cudaMemcpyHostToDevice), DEBUG_MEMCPY);
  cudaEventRecord(finish, 0);
  cudaEventSynchronize(finish);
  cudaEventElapsedTime(&timings[1], start, finish);

  // propagate
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid(1, (n + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
  cudaEventRecord(start, 0);
#ifndef SHARED_MEM
  propagate_row_heat_per_block<<<grid, block>>>(devA, devB, n, m, iterations);
#else
  propagate_row_heat_per_block_with_shared<<<grid, block>>>(devA, devB, n, m, iterations);
#endif
  CUDA_CHECK(cudaDeviceSynchronize(), DEBUG_FUNCTION);
  cudaEventRecord(finish, 0);
  cudaEventSynchronize(finish);
  cudaEventElapsedTime(&timings[2], start, finish);

  // average
  if (average)
  {
    dim3 b2(BLOCK_SIZE_X_AVG, BLOCK_SIZE_Y_AVG);
    dim3 g2(1, (n + BLOCK_SIZE_Y_AVG - 1) / BLOCK_SIZE_Y_AVG);
    cudaEventRecord(start, 0);
    average_row_heat<<<g2, b2, BLOCK_SIZE_Y_AVG * sizeof(FLOAT)>>>(devA, n, m, devAvg);
    CUDA_CHECK(cudaDeviceSynchronize(), DEBUG_FUNCTION);
    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(&timings[3], start, finish);
  }

  // copy out
  cudaEventRecord(start, 0);
  CUDA_CHECK(cudaMemcpy(matrix, devA, n * m * sizeof(FLOAT), cudaMemcpyDeviceToHost), DEBUG_MEMCPY);
  if (average)
    CUDA_CHECK(cudaMemcpy(averages, devAvg, n * sizeof(FLOAT), cudaMemcpyDeviceToHost), DEBUG_MEMCPY);
  cudaEventRecord(finish, 0);
  cudaEventSynchronize(finish);
  cudaEventElapsedTime(&timings[4], start, finish);

  // free
  cudaEventRecord(start, 0);
  CUDA_CHECK(cudaFree(devA), DEBUG_FREE);
  CUDA_CHECK(cudaFree(devB), DEBUG_FREE);
  if (average)
    CUDA_CHECK(cudaFree(devAvg), DEBUG_FREE);
  cudaEventRecord(finish, 0);
  cudaEventSynchronize(finish);
  cudaEventElapsedTime(&timings[5], start, finish);

  cudaEventDestroy(start);
  cudaEventDestroy(finish);
}