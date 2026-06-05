#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_mem.h"

float* cuda_malloc_float(int count){
    float* d_ptr;

    //cuda malloc writes a device pointer into the pointer we set
    //this pointer exists ONLY in the GPU Vram, so it does not exist in the eyes of the CPU
    //only usable on other CUDA/cuBLAS calls

    CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(float)));
    return d_ptr;
}

//frees a device pointer
void cuda_free(void* d_ptr){
    //again, only delete if it exists
    if(d_ptr){
        CUDA_CHECK(cudaFree(d_ptr));
    }
}


//copy count floats from host to device
//d_dst must be a device pointer from cuda_malloc_float (so it actually has mem to go to)
//h_src will he the host pointer

void cuda_copy_to_device(float* d_dst, const float* h_src, int count){

    //i love cuda and just them giving the funcs already <3
    //anyways, the direction flag tells CUDA which way the DMA transfer goes
    //this is a synchronous call, so it will block the CPU until the transfer is done
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, count * sizeof(float),
                          cudaMemcpyHostToDevice));
}

//backward op from GPU to CPU
void cuda_copy_to_host(float* h_dst, const float* d_src, int count) {
    //This is how you read results back from the GPU after a kernel runs.
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, count * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

//zero out count floats on the gpu, used to clear gradient buffers between our traning steps without having to go back to CPU just for that
void cuda_memset_zero(float* d_ptr, int count) {
    //byte by byte setting to 0
    CUDA_CHECK(cudaMemset(d_ptr, 0, count * sizeof(float)));
}