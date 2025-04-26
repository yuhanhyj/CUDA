#include "radiator.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

FLOAT row_iteration(FLOAT *matrix_old, int i, int j, int m);
static void perform_iteration(FLOAT *matrix_old, FLOAT *matrix_new, int n, int m);
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
        average_rows(matrix_a, n, m, averages);
        end_avg = clock();
    }
    timings_cpu[0] = (double)(end - start) / CLOCKS_PER_SEC;
    timings_cpu[1] = (double)(end_avg - start_avg) / CLOCKS_PER_SEC;
}

FLOAT row_iteration(FLOAT *matrix_old, int i, int j, int m)
{
    FLOAT res = 1.6f * matrix_old[i * m + (j - 2) % m] +
                1.55f * matrix_old[i * m + (j - 1) % m] +
                matrix_old[i * m + j % m] +
                0.6f * matrix_old[i * m + (j + 1) % m] +
                0.25f * matrix_old[i * m + (j + 2) % m];
    res *= 0.2f;
    return res;
}

static void perform_iteration(FLOAT *matrix_old, FLOAT *matrix_new, int n, int m)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 1; j < m - 1; ++j)
        {
            matrix_new[i * m + j] = row_iteration(matrix_old, i, j, m);
        }
    }
}

static void propagate_heat(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations)
{
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        perform_iteration(matrix_a, matrix_b, n, m);
        if (++iteration < iterations)
        {
            perform_iteration(matrix_b, matrix_a, n, m);
        }
        else
        {
            memcpy(matrix_a, matrix_b, n * m * sizeof(*matrix_a));
        }
    }
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