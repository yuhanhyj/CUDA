#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv)
{
    uint32_t n = 30;
    uint32_t m = 30;
    uint32_t iterations = 10;
    uint8_t average = 0;
    uint8_t print_time = 0;
    uint8_t skip_cpu = 0;
    for (int i = 1; i < argc; ++i)
    {
        char *arg = argv[i];
        if (arg[0] != '-' || arg[1] == '\0' || arg[2] != '\0')
        {
            fprintf(stderr, "Unrecognised argument: %s\n", arg);
            exit(EXIT_FAILURE);
        }
        switch (arg[1])
        {
        case 'm':
            if (i++ == argc - 1 || sscanf(argv[i], "%u", &m) != 1)
            {
                fprintf(stderr, "Expected integer after argument -m, but got: %s\n",
                        argv[i]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'n':
            if (i++ == argc - 1 || sscanf(argv[i], "%u", &n) != 1)
            {
                fprintf(stderr, "Expected integer after argument -n, but got: %s\n",
                        argv[i]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'p':
            if (i++ == argc - 1 || sscanf(argv[i], "%u", &iterations) != 1)
            {
                fprintf(stderr, "Expected integer after argument -p, but got: %s\n",
                        argv[i]);
                exit(EXIT_FAILURE);
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
            exit(EXIT_FAILURE);
            break;
        }
    }
    printf("size(%u, %u)\n\n", n, m);

    float *matrix_a = malloc(n * m * sizeof(*matrix_a));
    if (matrix_a == NULL)
    {
        fprintf(stderr, "Failed to allocate matrix_a\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; ++i)
    {
        matrix_a[i * m] = 0.98 * (float)((i + 1) * (i + 1)) / (float)(n * n);
        for (int j = 1; j < m; ++j)
        {
            matrix_a[i * m + j] = matrix_a[i * m] * ((m - j) * (m - j) / (m * m));
        }
    }

    float *matrix_cpu = malloc(n * m * sizeof(*matrix_cpu));
    if (matrix_cpu == NULL)
    {
        fprintf(stderr, "Failed to allocate matrix_cpu\n");
        exit(EXIT_FAILURE);
    }
    memcpy(matrix_cpu, matrix_a, n * m * sizeof(*matrix_a));

    float *matrix_cuda = malloc(n * m * sizeof(*matrix_cuda));
    if (matrix_cuda == NULL)
    {
        fprintf(stderr, "Failed to allocate matrix_cuda on cpu\n");
        exit(EXIT_FAILURE);
    }
    memcpy(matrix_cuda, matrix_a, n * m * sizeof(*matrix_a));

    // TODO: Implement C and CUDA code here

    //
    if (matrix_a == NULL)
    {
        free(matrix_a);
    }
    if (matrix_cpu == NULL)
    {
        free(matrix_cpu);
    }
    if (matrix_cuda == NULL)
    {
        free(matrix_cuda);
    }
    return 0;
}