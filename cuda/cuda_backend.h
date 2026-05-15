#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include <stdio.h>
#include <stdlib.h>

//macro calls, these are solely here for error checking and will act as wrappers, 
//if any of these fail, they will print out the file name, line number, and the error message
//They use __FILE__, __LINE__, and cudaGetErrorString to get the error message

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)


//cublas version
#define CUBLAS_CHECK(call)                                              \
    do {                                                                \
        cublasStatus_t stat = (call);                                   \
        if (stat != CUBLAS_STATUS_SUCCESS) {                            \
            fprintf(stderr, "cuBLAS error at %s:%d — status %d\n",     \
                    __FILE__, __LINE__, (int)stat);                     \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

void             cuda_backend_init(void);
void             cuda_backend_cleanup(void);
cublasHandle_t   cuda_backend_get_handle(void);

#ifdef __cplusplus
}
#endif

#endif
