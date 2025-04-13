#include "radiator.h"
#include "cuda_debug.h"

#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 1
#define BLOCK_SIZE_X_AVG 8
#define BLOCK_SIZE_Y_AVG 8

#define MAX_SHARED_MEM 8192

extern "C"
{
    __global__ void propagate_row_heat_per_block(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations);
    __global__ void propagate_row_heat_per_block_with_shared(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations);
    __global__ void average_row_heat(FLOAT *matrix, int n, int m, FLOAT *outputs);
}

void cuda_propagate_heat(FLOAT *matrix, int n, int m, int iterations, float *timings, FLOAT *averages, bool average)
{
    FLOAT *matrix_cuda_a;
    FLOAT *matrix_cuda_b;
    FLOAT *averages_cuda;

    cudaEvent_t start, finish;
    cudaEventCreate(&start);
    cudaEventCreate(&finish);

    cudaEventRecord(start, 0);

    CUDA_CHECK(cudaMalloc(&matrix_cuda_a, n * m * sizeof(*matrix_cuda_a)), DEBUG_MALLOC);
    CUDA_CHECK(cudaMalloc(&matrix_cuda_b, n * m * sizeof(*matrix_cuda_b)), DEBUG_MALLOC);

    if (average)
    {
        CUDA_CHECK(cudaMalloc(&averages_cuda, n * sizeof(*averages_cuda)), DEBUG_MALLOC);

        cudaEventRecord(finish, 0);
        cudaEventSynchronize(finish);
        cudaEventElapsedTime(timings, start, finish);
    }

    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(matrix_cuda_a, matrix, n * m * sizeof(*matrix), cudaMemcpyHostToDevice), DEBUG_MEMCPY);

    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(timings + 1, start, finish);

    CUDA_CHECK(cudaMemcpy(matrix_cuda_b, matrix_cuda_a, n * m * sizeof(*matrix), cudaMemcpyDeviceToDevice), DEBUG_MEMCPY);

    dim3 dim_block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dim_grid(1, n / BLOCK_SIZE_Y);

    cudaEventRecord(start, 0);

    propagate_row_heat_per_block<<<dim_grid, dim_block>>>(matrix_cuda_a, matrix_cuda_b, n, m, iterations);
    // propagate_row_heat_per_block_with_shared<<<dim_grid, dim_block>>>(matrix_cuda_a, matrix_cuda_b, n, m, iterations);
    CUDA_CHECK(cudaDeviceSynchronize(), DEBUG_FUNCTION);

    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(timings + 2, start, finish);

    if ((iterations % 2) == 1)
    {
        FLOAT *temp = matrix_cuda_a;
        matrix_cuda_a = matrix_cuda_b;
        matrix_cuda_b = temp;
    }
    if (average)
    {
        dim3 dim_block_avg(BLOCK_SIZE_X_AVG, BLOCK_SIZE_Y_AVG);
        dim3 dim_grid_avg(1, n / BLOCK_SIZE_Y_AVG);
        cudaEventRecord(start, 0);

        average_row_heat<<<dim_grid_avg, dim_block_avg>>>(matrix_cuda_a, n, m, averages_cuda);
        CUDA_CHECK(cudaDeviceSynchronize(), DEBUG_FUNCTION);

        cudaEventRecord(finish, 0);
        cudaEventSynchronize(finish);
        cudaEventElapsedTime(timings + 3, start, finish);
    }

    cudaEventRecord(start, 0);

    CUDA_CHECK(cudaMemcpy(matrix, matrix_cuda_a, n * m * sizeof(*matrix), cudaMemcpyDeviceToHost), DEBUG_MEMCPY);
    if (average)
    {
        CUDA_CHECK(cudaMemcpy(averages, averages_cuda, n * sizeof(*matrix), cudaMemcpyDeviceToHost), DEBUG_MEMCPY);
    }

    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(timings + 4, start, finish);

    cudaEventRecord(start, 0);

    CUDA_CHECK(cudaFree(matrix_cuda_a), DEBUG_FREE);
    CUDA_CHECK(cudaFree(matrix_cuda_b), DEBUG_FREE);
    if (average)
    {
        CUDA_CHECK(cudaFree(averages_cuda), DEBUG_FREE);
    }

    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(timings + 5, start, finish);
}

extern "C"
{
    __global__ void propagate_row_heat_per_block(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations)
    {
        return;
    }

    __global__ void propagate_row_heat_per_block_with_shared(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations)
    {
        return;
    }

    __global__ void average_row_heat(FLOAT *matrix, int n, int m, FLOAT *outputs)
    {
        return;
    }
}