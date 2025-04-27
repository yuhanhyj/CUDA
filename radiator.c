#include "radiator.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

static FLOAT row_iteration(const FLOAT *matrix_old, int i, int j, int m);
static void perform_iteration(const FLOAT *matrix_old, FLOAT *matrix_new, int n, int m);
static void propagate_heat(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations);
static void average_rows(FLOAT *matrix, int n, int m, FLOAT *averages);

void cpu_propagate_heat(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations, float *timings_cpu, FLOAT *averages, uint8_t average)
{
    clock_t start = 0;
    clock_t end = 0;
    clock_t start_avg = 0;
    clock_t end_avg = 0;

    start = clock();
    propagate_heat(matrix_a, matrix_b, n, m, iterations);
    end = clock();
    
    if (average)
    {
        start_avg = clock();
        // Use the final buffer that contains the result after propagation
        // This will be matrix_a if iterations is odd, matrix_b if iterations is even
        FLOAT *final_matrix = (iterations % 2 == 0) ? matrix_a : matrix_b;
        average_rows(final_matrix, n, m, averages);
        end_avg = clock();
    }
    
    timings_cpu[0] = (double)(end - start) / CLOCKS_PER_SEC;
    timings_cpu[1] = (double)(end_avg - start_avg) / CLOCKS_PER_SEC;
}

static FLOAT row_iteration(const FLOAT *matrix_old, int i, int j, int m)
{
    // Calculate all indices with modulo to handle wrap-around correctly
    int j_minus_2 = (j - 2 + m) % m;
    int j_minus_1 = (j - 1 + m) % m;
    int j_plus_1 = (j + 1) % m;
    int j_plus_2 = (j + 2) % m;
    
    FLOAT res = 1.6f * matrix_old[i * m + j_minus_2] +
                1.55f * matrix_old[i * m + j_minus_1] +
                matrix_old[i * m + j] +
                0.6f * matrix_old[i * m + j_plus_1] +
                0.25f * matrix_old[i * m + j_plus_2];
    
    return res * 0.2f;
}

static void perform_iteration(const FLOAT *matrix_old, FLOAT *matrix_new, int n, int m)
{
    for (int i = 0; i < n; ++i)
    {
        // Preserve the boundary values (first and last column)
        matrix_new[i * m] = matrix_old[i * m];
        matrix_new[i * m + (m - 1)] = matrix_old[i * m + (m - 1)];
        
        // Update interior points
        for (int j = 1; j < m - 1; ++j)
        {
            matrix_new[i * m + j] = row_iteration(matrix_old, i, j, m);
        }
    }
}

static void propagate_heat(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations)
{
    // Progress reporting threshold
    int report_threshold = iterations / 10;
    if (report_threshold == 0) report_threshold = 1;
    
    // Ping-pong buffering
    FLOAT *src = matrix_a;
    FLOAT *dst = matrix_b;
    
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        // Report progress for large matrices
        if (n > 4096 && iteration % report_threshold == 0 && iteration > 0) {
            printf("CPU iteration progress: %d%% complete\n", (iteration * 100) / iterations);
        }
        
        // Perform iteration from src to dst
        perform_iteration(src, dst, n, m);
        
        // Swap buffers for next iteration
        FLOAT *temp = src;
        src = dst;
        dst = temp;
    }
    
    // Ensure the final result is in the expected buffer (matrix_b)
    // If iterations is odd, result is in src (which is now pointing to matrix_b)
    // If iterations is even, result is in src (which is now pointing to matrix_a)
    if (iterations % 2 == 0)
    {
        // No need to copy, just swap the pointers
        // matrix_a already contains the final result, and matrix_b is unused
        // The caller expects the result in matrix_b, so we'll swap references
        memcpy(matrix_b, matrix_a, n * m * sizeof(FLOAT));
    }
    // If iterations is odd, the result is already in matrix_b, no action needed
}

static void average_rows(FLOAT *matrix, int n, int m, FLOAT *averages)
{
    for (int i = 0; i < n; ++i)
    {
        averages[i] = 0.0f;
        for (int j = 1; j < m - 1; ++j)
        {
            averages[i] += matrix[i * m + j];
        }
        averages[i] /= (m - 2);
    }
}