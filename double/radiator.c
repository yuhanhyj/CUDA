#include "radiator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Initialize a matrix with ghost columns for wrap-around handling
 * This adds two extra columns per row to eliminate expensive modulo operations
 */
static void initialize_matrix_with_ghost_columns(FLOAT *matrix, int n, int m, int padded_m)
{
    // For each row
    for (int i = 0; i < n; ++i)
    {
        // Compute base value for the row
        FLOAT base_value = 0.98f * (FLOAT)((i + 1) * (i + 1)) / (FLOAT)(n * n);
        
        // Fill main column values
        matrix[i * padded_m] = base_value; // First column
        for (int j = 1; j < m; ++j)
        {
            matrix[i * padded_m + j] = base_value * ((m - j) * (m - j)) / (FLOAT)(m * m);
        }
        
        // Add ghost columns (wrap-around values)
        matrix[i * padded_m + m] = matrix[i * padded_m]; // m+0 position gets value from column 0
        matrix[i * padded_m + m + 1] = matrix[i * padded_m + 1]; // m+1 position gets value from column 1
    }
}

/**
 * Process a single row for one iteration
 * This is highly optimized to avoid conditional branches and modulo operations
 */
static void process_row_iteration(FLOAT *dst_row, const FLOAT *src_row, int m)
{
    // Special case for column 1 (which needs values from ghost columns when looking left)
    {
        int j = 1;
        // Left neighbors accessed with wrap-around (using ghost columns)
        FLOAT old_l2 = src_row[m - 1];  // This is the wrapped value from the end
        FLOAT old_l1 = src_row[0];      // First column
        
        // Right neighbors (straightforward access)
        FLOAT old_r1 = src_row[j + 1];
        FLOAT old_r2 = src_row[j + 2];
        
        dst_row[j] = ((1.6f * old_l2) + (1.55f * old_l1) + src_row[j] + 
                     (0.6f * old_r1) + (0.25f * old_r2)) * 0.2f;
    }
    
    // Process the rest of the row (columns 2 to m-2)
    // No conditionals or modulo needed for these internal points
    for (int j = 2; j < m - 1; ++j)
    {
        dst_row[j] = ((1.6f * src_row[j - 2]) + (1.55f * src_row[j - 1]) +  src_row[j] + (0.6f * src_row[j + 1]) +  (0.25f * src_row[j + 2])) * 0.2f;
    }
    
    // Copy boundary values and update ghost columns for next iteration
    dst_row[0] = src_row[0];          // Preserve first column
    dst_row[m - 1] = src_row[m - 1];  // Preserve last column
    dst_row[m] = dst_row[0];          // Update first ghost column
    dst_row[m + 1] = dst_row[1];      // Update second ghost column
}

/**
 * Process multiple iterations for a single row
 * Processes pairs of iterations when possible for better efficiency
 */
static void process_row_iterations(FLOAT *row_a, FLOAT *row_b, int m, int iterations)
{
    // Handle odd number of iterations with an initial single step
    if (iterations % 2 != 0)
    {
        process_row_iteration(row_b, row_a, m);
        
        // Swap pointers for remaining iterations
        FLOAT *temp = row_a;
        row_a = row_b;
        row_b = temp;
        
        iterations--; // Reduce by one to get an even number
    }
    
    // Process remaining iterations in pairs for efficiency
    for (int iter = 0; iter < iterations / 2; ++iter)
    {
        // Two iterations at once with swapped pointers
        process_row_iteration(row_b, row_a, m);
        process_row_iteration(row_a, row_b, m);
    }
}

/**
 * Main heat propagation function with enhanced row-oriented processing
 */
static void propagate_heat(FLOAT *matrix_a, FLOAT *matrix_b, int n, int m, int iterations)
{
    // We'll add 2 extra columns per row for ghost/padding columns
    const int padded_m = m + 2;
    
    // Allocate temporary buffers with ghost columns
    FLOAT *padded_a = (FLOAT *)malloc(n * padded_m * sizeof(FLOAT));
    FLOAT *padded_b = (FLOAT *)malloc(n * padded_m * sizeof(FLOAT));
    
    if (!padded_a || !padded_b)
    {
        fprintf(stderr, "Failed to allocate memory for padded matrices\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize with ghost columns
    initialize_matrix_with_ghost_columns(padded_a, n, m, padded_m);
    
    // Progress reporting threshold
    int report_threshold = iterations > 10 ? iterations / 10 : 1;
    
    // Process each row independently
    for (int i = 0; i < n; ++i)
    {
        // Report progress for large matrices
        if (n > 4096 && i % (n/10) == 0)
        {
            printf("CPU processing row: %d/%d\n", i, n);
        }
        
        // Get pointers to the current row in each buffer
        FLOAT *row_a = padded_a + (i * padded_m);
        FLOAT *row_b = padded_b + (i * padded_m);
        
        // Process all iterations for this row
        process_row_iterations(row_a, row_b, m, iterations);
        
        // Copy final result back to the original matrix format
        // If iterations is even, final result is in row_a, otherwise in row_b
        FLOAT *final_row = (iterations % 2 == 0) ? row_a : row_b;
        
        // Copy back to original matrix, excluding ghost columns
        for (int j = 0; j < m; ++j)
        {
            matrix_b[i * m + j] = final_row[j];
        }
    }
    
    // Copy result to matrix_a if needed for consistency with the original interface
    memcpy(matrix_a, matrix_b, n * m * sizeof(FLOAT));
    
    // Free temporary buffers
    free(padded_a);
    free(padded_b);
}

/**
 * Calculate average values for each row
 */
static void average_rows(FLOAT *matrix, int n, int m, FLOAT *averages)
{
    for (int i = 0; i < n; ++i)
    {
        FLOAT sum = 0.0f;
        for (int j = 1; j < m - 1; ++j)
        {
            sum += matrix[i * m + j];
        }
        averages[i] = sum / (m - 2);
    }
}

/**
 * Main CPU implementation for heat propagation
 */
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
        average_rows(matrix_b, n, m, averages);
        end_avg = clock();
    }
    
    timings_cpu[0] = (double)(end - start) / CLOCKS_PER_SEC;
    timings_cpu[1] = (double)(end_avg - start_avg) / CLOCKS_PER_SEC;
}