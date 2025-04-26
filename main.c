#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "radiator_cuda.h"
#include "radiator.h"
extern void cuda_propagate_heat(FLOAT *matrix, int n, int m, int iterations, float *timings, FLOAT *averages, uint8_t average);
int main(int argc, char **argv)
{
    uint32_t n = 30;
    uint32_t m = 30;
    uint32_t iterations = 10;
    uint8_t average = 0;
    uint8_t print_time = 0;
    uint8_t skip_cpu = 0;
    float cpu_times[2] = {0};
    float gpu_times[6] = {0};

    for (int i = 1; i < argc; ++i)
    {
        char *arg = argv[i];
        if (arg[0] != '-' || arg[1] == '\0' || arg[2] != '\0')
        {
            fprintf(stderr, "Unrecognised argument: %s\n", arg);
            return 1;
        }
        switch (arg[1])
        {
        case 'm':
            if (i++ == argc - 1 || sscanf(argv[i], "%u", &m) != 1)
            {
                fprintf(stderr,
                        "Expected integer after argument -m, but got: %s\n",
                        argv[i]);
                return 1;
            }
            break;
        case 'n':
            if (i++ == argc - 1 || sscanf(argv[i], "%u", &n) != 1)
            {
                fprintf(stderr,
                        "Expected integer after argument -n, but got: %s\n",
                        argv[i]);
                return 1;
            }
            break;
        case 'p':
            if (i++ == argc - 1 || sscanf(argv[i], "%u", &iterations) != 1)
            {
                fprintf(stderr,
                        "Expected integer after argument -p, but got: %s\n",
                        argv[i]);
                return 1;
            }
            break;
        case 't':
            print_time = 1;
            break;
        case 'a':
            average = 1;
            break;
        case 'c':
            skip_cpu = 1;
            break;
        default:
            fprintf(stderr, "Unrecognised argument: %s\n", arg);
            return 1;
            break;
        }
    }
    printf("size(%u, %u)\n\n", n, m);

    FLOAT *matrix_a = (FLOAT *)malloc(n * m * sizeof(*matrix_a));
    if (matrix_a == NULL)
    {
        fprintf(stderr, "Failed to allocate matrix_a\n");
        return 1;
    }

    for (int i = 0; i < n; ++i)
    {
        matrix_a[i * m] = 0.98 * (FLOAT)((i + 1) * (i + 1)) / (FLOAT)(n * n);
        for (int j = 1; j < m; ++j)
        {
            matrix_a[i * m + j] =
                matrix_a[i * m] * ((m - j) * (m - j) / (m * m));
        }
    }
    FLOAT *matrix_cpu = NULL;
    if (!skip_cpu)
    {
        matrix_cpu = (FLOAT *)malloc(n * m * sizeof(*matrix_cpu));
        if (matrix_cpu == NULL)
        {
            fprintf(stderr, "Failed to allocate matrix_cpu\n");
            return 1;
        }
        memcpy(matrix_cpu, matrix_a, n * m * sizeof(*matrix_a));
    }

    FLOAT *matrix_cuda = (FLOAT *)malloc(n * m * sizeof(*matrix_cuda));
    if (matrix_cuda == NULL)
    {
        fprintf(stderr, "Failed to allocate matrix_cuda on cpu\n");
        return 1;
    }
    memcpy(matrix_cuda, matrix_a, n * m * sizeof(*matrix_a));

    // TODO: Implement C and CUDA code here
    FLOAT *averages = (FLOAT *)malloc(n * sizeof(*averages));
    if (!skip_cpu)
    {
        cpu_propagate_heat(matrix_a, matrix_cpu, n, m, iterations, cpu_times,averages, average);
    }
    FLOAT *cuda_averages = (FLOAT *)malloc(n * sizeof(*cuda_averages));
    cuda_propagate_heat(matrix_cuda, n, m, iterations, gpu_times, cuda_averages,average);
    if (print_time)
    {
        printf("--------timing info--------\n");
        printf("GPU:\nMalloc: %lfms\n", gpu_times[0]);
        printf("Copy to Device: %lfms\n", gpu_times[1]);
        printf("Propagation: %lfms\n", gpu_times[2]);
        if (average)
        {
            printf("Averaging: %lfms\n", gpu_times[3]);
        }
        printf("Copy to Host: %lfms\n", gpu_times[4]);
        printf("Free: %lfms\n", gpu_times[5]);
        if (average)
        {
            printf("Total: %lfms\n", gpu_times[0] + gpu_times[1] + gpu_times[2] + gpu_times[3] + gpu_times[4] + gpu_times[5]);
        }
        else
        {
            printf("Total: %lfms\n", gpu_times[0] + gpu_times[1] +gpu_times[2] + gpu_times[4] + gpu_times[5]);
        }
        if (!skip_cpu)
        {
            printf("\n\nCPU:\n");
            printf("Propagation: %lfs\n", (double)cpu_times[0]);
            if (average)
            {
                printf("Averaging: %lfs\n", (double)cpu_times[1]);
                printf("Total: %lfs\n",
                       (double)cpu_times[0] + (double)cpu_times[1]);
            }
            printf("\nSpeedups:\n");
            printf("Propataion: %lf\n",
                   (double)cpu_times[0] / (gpu_times[2] / 1000));
            if (average)
            {
                printf("Averaging: %lf\n",
                       (double)cpu_times[1] / (gpu_times[3] / 1000));
                printf("Overall (Excluding transfers): %lf\n",
                       ((double)cpu_times[1] + (double)cpu_times[0]) /
                           ((gpu_times[3] + gpu_times[2]) / 1000));
                printf("Overall (Including transfers): %lf\n",
                       ((double)cpu_times[1] + (double)cpu_times[0]) /
                           ((gpu_times[0] + gpu_times[1] + gpu_times[2] + gpu_times[3] +
                             gpu_times[4] + gpu_times[5]) /
                            1000));
            }
            else
            {
                printf("Overall (Excluding transfers): %lf\n",
                       (double)cpu_times[0] / ((gpu_times[2]) / 1000));
                printf("Overall (Including transfers): %lf\n",
                       (double)cpu_times[0] / ((gpu_times[0] + gpu_times[1] + gpu_times[2] + gpu_times[4] + gpu_times[5]) / 1000));
            }
        }
    }
    if (!skip_cpu)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if (fabs(matrix_a[i * m + j] - matrix_cuda[i * m + j]) > 1e-4)
                {
                    fprintf(stderr,
                            "Error: matrix_a[%d][%d] = %f, matrix_cuda[%d][%d] "
                            "= %f\n",
                            i, j, matrix_a[i * m + j], i, j,
                            matrix_cuda[i * m + j]);
                    return 1;
                }
            }
        }
        if (average)
        {
            for (int i = 0; i < n; ++i)
            {
                if (fabs(averages[i] - cuda_averages[i]) > 1e-4)
                {
                    fprintf(
                        stderr,
                        "Error: averages[%d] = %f, cuda_averages[%d] = %f\n", i,
                        averages[i], i, cuda_averages[i]);
                    return 1;
                }
            }
        }
    }

    //
    if (matrix_a != NULL)
    {
        free(matrix_a);
    }
    if (matrix_cpu != NULL)
    {
        free(matrix_cpu);
    }
    if (matrix_cuda != NULL)
    {
        free(matrix_cuda);
    }
    if (averages != NULL)
    {
        free(averages);
    }
    if (cuda_averages != NULL)
    {
        free(cuda_averages);
    }
    return 0;
}