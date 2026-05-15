//we want to mark any op that runs on the gpu as __global__ so the compiler knos to compile it for the gpu not cpu
//then when laucned, we need to specify threads and blocks. threads typically are grouped as blocks of 256 or 512
//we just need to calc how many blocks wed need

//so were gonna see <<<block, threads>>> syntax between func names as args for CUDA

#ifndef ELEMENTWISE_CUH
#define ELEMENTWISE_CUH

#include "../cuda_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

//funcs calls would look like: intput1, intput2, output, count of floats

//out[i] = a[i] + b[i] for count elements, basic sum
void cuda_add(const float* d_a, const float* d_b, float* d_out, int count);

//out[i] = a[i] - b[i], basic diff
void cuda_sub(const float* d_a, const float* d_b, float* d_out, int count);

//out[i] = a[i] * b[i] hadamard, not dot
void cuda_mul(const float* d_a, const float* d_b, float* d_out, int count);

// out[i] = a[i] * scalar 
void cuda_scale(const float* d_a, float scalar, float* d_out, int count);

//activation funcs

// out[i] = max(0, a[i]), (RELU)
void cuda_relu_forward(const float* d_a, float* d_out, int count);

// out[i] = 1 / (1 + exp(-a[i])) (SIGMOID)
void cuda_sigmoid_forward(const float* d_a, float* d_out, int count);

// out[i] = tanh(a[i])
void cuda_tanh_forward(const float* d_a, float* d_out, int count);


//activation derivatives

// out[i] = dA[i] * (a[i] > 0 ? 1 : 0)
//a is the pre activation input, z is not the output here
//dA is upstream gradient flowing back (effectively new incoming gradient)
void cuda_relu_backward(const float* d_a, const float* d_dA, float* d_out, int count);

// out[i] = dA[i] * sigmoid(a[i]) * (1 - sigmoid(a[i]))
// a is the pre-activation input Z.
void cuda_sigmoid_backward(const float* d_a, const float* d_dA, float* d_out, int count);

//out[i] = dA[i] * (1 - tanh(a[i])^2)
//a is the pre-activation input Z. 
void cuda_tanh_backward(const float* d_a, const float* d_dA, float* d_out, int count);


#ifdef __cplusplus
}
#endif

#endif