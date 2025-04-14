#ifndef _CUDA_DEBUG_H_
#define _CUDA_DEBUG_H_

#include <cuda_runtime.h>
#include <stdio.h>

enum DEBUG_LEVEL
{
    DEBUG_NONE,
    DEBUG_MALLOC,
    DEBUG_MEMCPY,
    DEBUG_FUNCTION,
    DEBUG_FREE,
    DEBUG_ERROR,
};

int DEBUG_THREAD = DEBUG_ERROR;

#define CUDA_CHECK(call, flag)                                             \
    do                                                                     \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess && flag <= DEBUG_THREAD)                    \
        {                                                                  \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                  \
            printf("code:%d, reason: %s\n", err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#endif // _CUDA_DEBUG_H_