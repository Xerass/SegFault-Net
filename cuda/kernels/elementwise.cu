#include "elementwise.cuh"

//256 threads per block
//GPU blocks can reach 1024 but 256 is more common for more devices
#define BLOCK_SIZE 256

//for every func wed want to create a grid-stride loop pattern
//this is a common CUDA pattern to allow for any size of input, 
//even those larger than our total thread count, by having each thread loop over 
//multiple elements in a strided way

//arithmetic kernels

__global__ void kernel_add(const float* a, const float* b, float* out, int n){
    // blocIdx.x is the blox index
    // blocdim.x = threads PER block
    // threadID.x = thread pos within the block (so for thsi its 0-255)

    //idx is just a unique global thread index across the whole grid, so we can use it to index into our arrays
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i+= stride){
        out[i] = a[i] + b[i];
    }
}

//everything else is basically the same just with diff math ops
__global__ void kernel_sub(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] - b[i];
    }
}

__global__ void kernel_mul(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}

__global__ void kernel_scale(const float* a, float scalar, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] * scalar;
    }
}

__global__ void kernel_relu_forward(const float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

__global__ void kernel_sigmoid_forward(const float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = 1.0f / (1.0f + expf(-a[i]));
    }
}

__global__ void kernel_tanh_forward(const float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = tanhf(a[i]);
    }
}

//derivative of acts are a little different though
//for each it will take a:
/*

a = pre activation input Z (cached from forward)
dA = upstream gradient flowing back (prev gradient incoming)
out = same as before

This is the element-wise part of the chain rule:
    dZ = dA * g'(Z)
where g' is the activation derivative and * is Hadamard product.
*/

__global__ void kernel_relu_backward(const float* a, const float* dA, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        // ReLU derivative: 1 if input > 0, else 0.
          // So gradient passes through unchanged where active, dies where inactive.
        out[i] = a[i] > 0.0f ? dA[i] : 0.0f;
    }
}


__global__ void kernel_sigmoid_backward(const float* a, const float* dA, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        // sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        // Recompute sigmoid from Z rather than storing it, trades compute for memory.
        float s = 1.0f / (1.0f + expf(-a[i]));
        out[i] = dA[i] * s * (1.0f - s);
    }
}

__global__ void kernel_tanh_backward(const float* a, const float* dA, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        // tanh'(z) = 1 - tanh(z)^2
        // Same tradeoff as sigmoid: recompute rather than store.
        float t = tanhf(a[i]);
        out[i] = dA[i] * (1.0f - t * t);
    }
}


// NOTE, since the lib is gonna be in C, we can directly inject  <<<block, threads >>>>
//were gonna have to create a public facing wrapper function for each kernel that will be called from the C interface, and in those wrapper functions we can set up the grid and block dimensions and launch the kernels.


// helper to calc block sizes needed to cover n elements
//not just normal divide since we need to insert integer ceiling division
static int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


//actual funcs we can call from C
void cuda_add(const float* d_a, const float* d_b, float* d_out, int count) {
    kernel_add<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_b, d_out, count);
}

void cuda_sub(const float* d_a, const float* d_b, float* d_out, int count) {
    kernel_sub<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_b, d_out, count);
}

void cuda_mul(const float* d_a, const float* d_b, float* d_out, int count) {
    kernel_mul<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_b, d_out, count);
}

void cuda_scale(const float* d_a, float scalar, float* d_out, int count) {
    kernel_scale<<<grid_size(count), BLOCK_SIZE>>>(d_a, scalar, d_out, count);
}

void cuda_relu_forward(const float* d_a, float* d_out, int count) {
    kernel_relu_forward<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_out, count);
}

void cuda_sigmoid_forward(const float* d_a, float* d_out, int count) {
    kernel_sigmoid_forward<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_out, count);
}

void cuda_tanh_forward(const float* d_a, float* d_out, int count) {
    kernel_tanh_forward<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_out, count);
}

void cuda_relu_backward(const float* d_a, const float* d_dA, float* d_out, int count) {
    kernel_relu_backward<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_dA, d_out, count);
}

void cuda_sigmoid_backward(const float* d_a, const float* d_dA, float* d_out, int count) {
    kernel_sigmoid_backward<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_dA, d_out, count);
}

void cuda_tanh_backward(const float* d_a, const float* d_dA, float* d_out, int count) {
    kernel_tanh_backward<<<grid_size(count), BLOCK_SIZE>>>(d_a, d_dA, d_out, count);
}