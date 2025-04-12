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

    return 0;
}