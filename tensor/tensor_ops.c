#include "tensor_ops.h"
#include "../cuda/kernels/elementwise.cuh"
#include "../cuda/kernels/reduce.cuh"
#include "../cuda/kernels/matmul.cuh"
#include "../cuda/kernels/conv2d.cuh"
#include "../cuda/kernels/batchnorm.cuh"

void tensor_add(const Tensor* a, const Tensor* b, Tensor* out) {
    cuda_add(a->data, b->data, out->data, a->size);
}

void tensor_sub(const Tensor* a, const Tensor* b, Tensor* out) {
    cuda_sub(a->data, b->data, out->data, a->size);
}

void tensor_mul(const Tensor* a, const Tensor* b, Tensor* out) {
    cuda_mul(a->data, b->data, out->data, a->size);
}

void tensor_scale(const Tensor* a, float scalar, Tensor* out) {
    cuda_scale(a->data, scalar, out->data, a->size);
}

void tensor_matmul(const Tensor* A, const Tensor* B, Tensor* out,
                   int M, int K, int N) {
    cuda_matmul(A->data, B->data, out->data, M, K, N);
}

void tensor_matmul_transA(const Tensor* A, const Tensor* B, Tensor* out,
                          int M, int K, int N) {
    cuda_matmul_transA(A->data, B->data, out->data, M, K, N);
}

void tensor_matmul_transB(const Tensor* A, const Tensor* B, Tensor* out,
                          int M, int K, int N) {
    cuda_matmul_transB(A->data, B->data, out->data, M, K, N);
}

void tensor_relu_forward(const Tensor* a, Tensor* out) {
    cuda_relu_forward(a->data, out->data, a->size);
}

void tensor_sigmoid_forward(const Tensor* a, Tensor* out) {
    cuda_sigmoid_forward(a->data, out->data, a->size);
}

void tensor_tanh_forward(const Tensor* a, Tensor* out) {
    cuda_tanh_forward(a->data, out->data, a->size);
}

void tensor_relu_backward(const Tensor* a, const Tensor* dA, Tensor* out) {
    cuda_relu_backward(a->data, dA->data, out->data, a->size);
}

void tensor_sigmoid_backward(const Tensor* a, const Tensor* dA, Tensor* out) {
    cuda_sigmoid_backward(a->data, dA->data, out->data, a->size);
}

void tensor_tanh_backward(const Tensor* a, const Tensor* dA, Tensor* out) {
    cuda_tanh_backward(a->data, dA->data, out->data, a->size);
}

void tensor_sum_rows(const Tensor* in, Tensor* out, int rows, int cols) {
    cuda_sum_rows(in->data, out->data, rows, cols);
}

void tensor_sum_cols(const Tensor* in, Tensor* out, int rows, int cols) {
    cuda_sum_cols(in->data, out->data, rows, cols);
}

void tensor_im2col(const Tensor* in, Tensor* col,
                   int C_in, int H, int W,
                   int kH, int kW, int stride, int pad) {
    cuda_im2col(in->data, C_in, H, W, kH, kW, stride, pad, col->data);
}

void tensor_col2im(const Tensor* col, Tensor* out,
                   int C_in, int H, int W,
                   int kH, int kW, int stride, int pad) {
    cuda_col2im(col->data, C_in, H, W, kH, kW, stride, pad, out->data);
}

void tensor_batchnorm_forward(const Tensor* in, Tensor* out,
                              const Tensor* gamma, const Tensor* beta,
                              Tensor* mean, Tensor* var,
                              int features, int batch_size, float epsilon) {
    cuda_batchnorm_forward(in->data, out->data,
                           gamma->data, beta->data,
                           mean->data, var->data,
                           features, batch_size, epsilon);
}

void tensor_batchnorm_backward(const Tensor* dout, const Tensor* in,
                               const Tensor* mean, const Tensor* var,
                               const Tensor* gamma,
                               Tensor* din, Tensor* dgamma, Tensor* dbeta,
                               int features, int batch_size, float epsilon) {
    cuda_batchnorm_backward(dout->data, in->data,
                            mean->data, var->data, gamma->data,
                            din->data, dgamma->data, dbeta->data,
                            features, batch_size, epsilon);
}