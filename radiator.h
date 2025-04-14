#ifndef _RADIATOR_H_
#define _RADIATOR_H_

#include "precision.h"

void cuda_propagate_heat(FLOAT *matrix, int n, int m, int iterations, float *timings, FLOAT *averages, uint8_t average);
void cpu_propagate_heat(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations, float *timings_cpu, FLOAT *averages, uint8_t average);
void average_rows(float *matrix, int n, int m, float *averages);

#endif // _RADIATOR_H_