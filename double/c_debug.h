#ifndef _C_DEBUG_H_
#define _C_DEBUG_H_

#include <stdio.h>

#define CHECK_PTR(ptr)                                                          \
    do                                                                          \
    {                                                                           \
        if (ptr == NULL)                                                        \
        {                                                                       \
            fprintf(stderr, "%s(%d)Error: NULL pointer\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                 \
        }
}
#endif // _C_DEBUG_H_