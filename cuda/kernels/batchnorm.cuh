#ifndef BATCHNORM_CUH
#define BATCHNORM_CUH

#include "../cuda_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

//batch norm kernels, 
// BN normalizes each feature / channel across the batch dims
// x_hat = (x - mean) / sqrt(variance + epsilon)
// y = gamma * x_hat + beta
// gamma determines scale, beta the shift are both learnable params
// epsilong just prevents div by 0


//batchnorm forward is straightfoward
//take in, out, gamma, beta, mean and var to calc x_hat
//we save d_mean and d_var for the backward pass
void cuda_batchnorm_forward(const float* d_in, float* d_out,
                            const float* d_gamma, const float* d_beta,
                            float* d_mean, float* d_var,
                            int features, int batch_size, float epsilon);
                            
//backward has d out, in, mean, var, gamma, outputs d in, d gamme and de beta
void cuda_batchnorm_backward(const float* d_dout, const float* d_in,
                             const float* d_mean, const float* d_var,
                             const float* d_gamma,
                             float* d_din, float* d_dgamma, float* d_dbeta,
                             int features, int batch_size, float epsilon);

#ifdef __cplusplus
}
#endif

#endif