#ifndef CUDA_MEM_H
#define CUDA_MEM_H


#include "cuda_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

//allocate counts floats on GPU VRAM, returns a device poiinter to the allocated memory
//note that we cannot dereference this pointer via the CPU, we can only use it in CUDA kernels
float* cuda_malloc_float(int count);
void cuda_free(void* d_ptr);
void cuda_copy_to_device(float* d_dst, const float* h_src, int count);
void cuda_copy_to_host(float* h_dst, const float* d_src, int count);
void cuda_memset_zero(float* d_ptr, int count);

#ifdef __cplusplus
}
#endif

#endif