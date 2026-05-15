//serves as the lowest layer of this lib
//will take on the cuda backend

//cuBLAS is the Linear Algebra Library for CUDA
#include <cublas_v2.h>

//for error wrappers
#include "cuda_backend.h"

// cublas handle, context object that cublas needs for every call
//its going to hold internal state stuff like workspace mem, the GPU target, data stream config

//need it to be static so its only visible here and cannot be called by other files
//everything else will just access it via the cuda_backend_get_handle for safety, one owner, one lifetime

//set this later
static cublasHandle_t g_cublas_handle = NULL;


void cuda_backend_init(void){
    //ask the CUDA driver how many gpus are visible

    int dev_count = 0;
    
    //first wrapper use <3 we spam this so we have clear error logs
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));

    if (dev_count == 0){
        fprintf(stderr, "No CUDA devices available!\n");
        exit(EXIT_FAILURE);
    }

    //if there are we need to bind the current CPU thread this program is active on to GPU 0
    //all later calls will now target that device

    CUDA_CHECK(cudaSetDevice(0));


    //just print out hardware specs, basic sanity check on startup
    //StreamMultiprocessor version determines what features/instructions
    //a gpu can actually support, multriprocessor count determines how many cores it has, and the total global memory is just a sanity check to make sure we can actually run our workloads on it

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    //print the relevant info
    printf("GPU: %s | SM %d.%d | %.0f MB VRAM | %d SMs\n",
           prop.name,
           prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0),
           prop.multiProcessorCount);

    //next we need to actually create a cublas context and assign it to our handle
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
}

cublasHandle_t cuda_backend_get_handle(void){
    //just return the handle, we only have one so no need for any fancy logic here
    return g_cublas_handle;
}

//clean up
void cuda_backend_cleanup(void){
    //just check if it exists first
    if (g_cublas_handle != NULL){
        CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
        g_cublas_handle = NULL;
    }
}