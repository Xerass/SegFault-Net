#ifndef CONV2D_CUH
#define CONV2D_CUH

#include "../cuda_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


//theres an interesting trick to "skip" having to do convolution
// we use im2col / col2im
// so instead of sliding a kernel across an image with nested loops
// im2col unrolls every pach the kernel would have touched into a column
// this just makes one big matrix for one big matmul
// col2im is its reverse used in backprop

// highkey just stole this and trusted the process


void cuda_im2col(const float* d_in,
                 int C_in, int H, int W,
                 int kH, int kW,
                 int stride, int pad,
                 float* d_col);

void cuda_col2im(const float* d_col,
                 int C_in, int H, int W,
                 int kH, int kW,
                 int stride, int pad,
                 float* d_out);
                 

#ifdef __cplusplus
}
#endif

#endif