#ifndef _RADIATOR_CUDA_H_
#define _RADIATOR_CUDA_H_

#include "precision.h"

#ifdef __cplusplus
extern "C"
{
    void cuda_propagate_heat(FLOAT *matrix, int n, int m, int iterations, float *timings, FLOAT *averages, uint8_t average);
}
#endif

#endif