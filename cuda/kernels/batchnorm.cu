#include "batchnorm.cuh"

#define BLOCK_SIZE 256

//forward kernel one thread per feat

__global__ void kernel_batchnorm_forward(const float* in, float* out,
                                         const float* gamma, const float* beta,
                                         float* mean, float* var,
                                         int features, int batch_size,
                                         float epsilon) {



    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int feat = f; feat < features; feat += stride){
        // mean = 1/B * sum(x) across batch for this feat
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += in[feat * batch_size + b];
        }
        float m = sum / batch_size;
        mean[feat] = m;

        //var 1/B * sum(x - mean)**2
        float vsum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            float diff = in[feat * batch_size + b] - m;
            vsum += diff * diff;
        }
        float v = vsum / batch_size;
        var[feat] = v;


        //nnormalize, scale, shift
        float inv_std = 1.0f / sqrtf(v + epsilon);
        for (int b = 0; b < batch_size; b++) {
            float x_hat = (in[feat * batch_size + b] - m) * inv_std;
            out[feat * batch_size + b] = gamma[feat] * x_hat + beta[feat];
    }
}
 }

//backward batchnorm kernel is messy
//since we only have ONE global mean and variance
//EVERY gradient for each sample DEPENDS on all other samples too

//dgamma = sum(dout * x_hat) across batch determines how much to adjust the scale
//dbeta = sum(dout) across batch determines how much to adjust the shift 
// din = (gamma / std) * (dout - dbeta/B - x_hat * dgamma/B)  -The direct gradient we use


//last tow terms tackle this batch coupling, they account for the fact that one input affects the global
//which affects the norm of eVERY other layer
__global__ void kernel_batchnorm_backward(const float* dout, const float* in,
                                          const float* mean, const float* var,
                                          const float* gamma,
                                          float* din, float* dgamma, float* dbeta,
                                          int features, int batch_size,
                                          float epsilon) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int feat = f; feat < features; feat += stride) {
        float m = mean[feat];
        float v = var[feat];
        float inv_std = 1.0f / sqrtf(v + epsilon);

        float dg = 0.0f;
        float db = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            int idx = feat * batch_size + b;
            float x_hat = (in[idx] - m) * inv_std;
            dg += dout[idx] * x_hat;
            db += dout[idx];
        }
        dgamma[feat] = dg;
        dbeta[feat] = db;

        float scale = gamma[feat] * inv_std / batch_size;
        for (int b = 0; b < batch_size; b++) {
            int idx = feat * batch_size + b;
            float x_hat = (in[idx] - m) * inv_std;
            din[idx] = scale * (batch_size * dout[idx] - db - x_hat * dg);
        }
    }
}


static int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


void cuda_batchnorm_forward(const float* d_in, float* d_out,
                            const float* d_gamma, const float* d_beta,
                            float* d_mean, float* d_var,
                            int features, int batch_size, float epsilon) {
    kernel_batchnorm_forward<<<grid_size(features), BLOCK_SIZE>>>(
        d_in, d_out, d_gamma, d_beta, d_mean, d_var,
        features, batch_size, epsilon);
}

void cuda_batchnorm_backward(const float* d_dout, const float* d_in,
                             const float* d_mean, const float* d_var,
                             const float* d_gamma,
                             float* d_din, float* d_dgamma, float* d_dbeta,
                             int features, int batch_size, float epsilon) {
    kernel_batchnorm_backward<<<grid_size(features), BLOCK_SIZE>>>(
        d_dout, d_in, d_mean, d_var, d_gamma,
        d_din, d_dgamma, d_dbeta,
        features, batch_size, epsilon);
}