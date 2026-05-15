#ifndef REDUCE_CUH
#define REDUCE_CUH

#include "../cuda_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

// sums the rows of row x col matrix
// input d_in is row major
// output d has rows elements
// we mainly use this for bias gradients, which are just the sum of the output
// gradients across the batch dimension, 
//so we can just use this to sum across the batch dimension for each neuron to get the bias gradient
void cuda_sum_rows(const float* d_in, float* d_out, int rows, int cols);


// sum each col of a row x col matrix
// used later down the line if we want to figure out PER FEATURE stats across a batch
void cuda_sum_cols(const float* d_in, float* d_out, int rows, int cols);


#ifdef __cplusplus
}
#endif

#endif