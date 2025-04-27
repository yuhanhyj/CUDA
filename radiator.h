#ifndef _RADIATOR_H_
#define _RADIATOR_H_

#include "precision.h"
#include <stdint.h>

/**
 * Propagates heat through a matrix using the CPU
 * 
 * This implementation uses:
 * - Row-oriented processing for better cache efficiency
 * - Ghost columns to eliminate expensive modulo operations
 * - Pair-processing of iterations where possible
 * - Optimized memory access patterns
 * 
 * @param matrix_a First buffer for computation (contains initial values and will be updated)
 * @param matrix_b Second buffer for computation (will contain final result)
 * @param n Number of rows in the matrix
 * @param m Number of columns in the matrix
 * @param iterations Number of iterations to perform
 * @param timings_cpu Array to store timing information [propagation_time, averaging_time]
 * @param averages Array to store row averages (if average flag is set)
 * @param average Flag indicating whether to calculate row averages
 */
void cpu_propagate_heat(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations, float *timings_cpu, FLOAT *averages, uint8_t average);

#endif // _RADIATOR_H_