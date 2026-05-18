#include "conv2d.cuh"

#define BLOCK_SIZE 256  

__global__ void kernel_im2col(const float* in,
                              int C_in, int H, int W,
                              int kH, int kW,
                              int stride, int pad,
                              int out_H, int out_W,
                              float* col) {
    
    
    int total = C_in * kH * kW * out_H * out_W;

    //tyipcal stuff
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    
    //the decoding stuff that i fully trusts works
    for (int i = idx; i < total; i += grid_stride) {

        //transform flat indexes into its actual 5D indexes
        int ow = i % out_W;
        int tmp = i / out_W;
        int oh = tmp % out_H;
        tmp = tmp / out_H;
        int kw = tmp % kW;
        tmp = tmp / kW;
        int kh = tmp % kH;
        int c  = tmp / kH;

        //map from output pos
        int ih = oh * stride - pad + kh;
        int iw = ow * stride - pad + kw;

        //bnounds check
        float val = 0.0f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            val = in[c * H * W + ih * W + iw];
        }

        col[i] = val;
    }                    

}

__global__ void kernel_col2im(const float* col,
                              int C_in, int H, int W,
                              int kH, int kW,
                              int stride, int pad,
                              int out_H, int out_W,
                              float* out) {
    int total = C_in * kH * kW * out_H * out_W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += grid_stride) {
        int ow = i % out_W;
        int tmp = i / out_W;
        int oh = tmp % out_H;
        tmp = tmp / out_H;
        int kw = tmp % kW;
        tmp = tmp / kW;
        int kh = tmp % kH;
        int c  = tmp / kH;

        int ih = oh * stride - pad + kh;
        int iw = ow * stride - pad + kw;

        //additional accumulation step, accumulate if were inside the actual image and not padding
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            atomicAdd(&out[c * H * W + ih * W + iw], col[i]);
        }
    }
}

static int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


void cuda_im2col(const float* d_in,
                 int C_in, int H, int W,
                 int kH, int kW,
                 int stride, int pad,
                 float* d_col) {
    //compute output spatial dimensions
    //standard convolution formula: out = (in + 2*pad - kernel) / stride + 1 */
    int out_H = (H + 2 * pad - kH) / stride + 1;
    int out_W = (W + 2 * pad - kW) / stride + 1;
    int total = C_in * kH * kW * out_H * out_W;

    kernel_im2col<<<grid_size(total), BLOCK_SIZE>>>(
        d_in, C_in, H, W, kH, kW, stride, pad, out_H, out_W, d_col);
}

void cuda_col2im(const float* d_col,
                 int C_in, int H, int W,
                 int kH, int kW,
                 int stride, int pad,
                 float* d_out) {
    int out_H = (H + 2 * pad - kH) / stride + 1;
    int out_W = (W + 2 * pad - kW) / stride + 1;
    int total = C_in * kH * kW * out_H * out_W;

    // d_out MUST be zeroed before calling thiss ince col2im accumulates
    //with atomicAdd, so leftover values would corrupt the gradient.
    kernel_col2im<<<grid_size(total), BLOCK_SIZE>>>(
        d_col, C_in, H, W, kH, kW, stride, pad, out_H, out_W, d_out);
}