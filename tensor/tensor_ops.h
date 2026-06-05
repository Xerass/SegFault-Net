
#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.h"

//all of these are just wrappers around the cuda kernels
void tensor_add(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_sub(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_mul(const Tensor* a, const Tensor* b, Tensor* out);
void tensor_scale(const Tensor* a, float scalar, Tensor* out);

void tensor_matmul(const Tensor* A, const Tensor* B, Tensor* out,
                   int M, int K, int N);
void tensor_matmul_transA(const Tensor* A, const Tensor* B, Tensor* out,
                          int M, int K, int N);
void tensor_matmul_transB(const Tensor* A, const Tensor* B, Tensor* out,
                          int M, int K, int N);

void tensor_relu_forward(const Tensor* a, Tensor* out);
void tensor_sigmoid_forward(const Tensor* a, Tensor* out);
void tensor_tanh_forward(const Tensor* a, Tensor* out);

void tensor_relu_backward(const Tensor* a, const Tensor* dA, Tensor* out);
void tensor_sigmoid_backward(const Tensor* a, const Tensor* dA, Tensor* out);
void tensor_tanh_backward(const Tensor* a, const Tensor* dA, Tensor* out);

void tensor_sum_rows(const Tensor* in, Tensor* out, int rows, int cols);
void tensor_sum_cols(const Tensor* in, Tensor* out, int rows, int cols);

void tensor_im2col(const Tensor* in, Tensor* col,
                   int C_in, int H, int W,
                   int kH, int kW, int stride, int pad);
void tensor_col2im(const Tensor* col, Tensor* out,
                   int C_in, int H, int W,
                   int kH, int kW, int stride, int pad);

void tensor_batchnorm_forward(const Tensor* in, Tensor* out,
                              const Tensor* gamma, const Tensor* beta,
                              Tensor* mean, Tensor* var,
                              int features, int batch_size, float epsilon);
void tensor_batchnorm_backward(const Tensor* dout, const Tensor* in,
                               const Tensor* mean, const Tensor* var,
                               const Tensor* gamma,
                               Tensor* din, Tensor* dgamma, Tensor* dbeta,
                               int features, int batch_size, float epsilon);

#endif