#include "matmul.cuh"
#include <cublas_v2.h>

//so, cublasSgemm is confusing as hell
// it assume COLUME major (old ass Fortran Convention)
// our matrices are all in row major for ease of understanding
// the fix is relatively simple

// a row-major matrix A(M,K) looks like a column-major matrix A^T(K,M)
//given tat we dont move any data in the process
// so what were effectively gonna do is multiply everything as its transpose
// along with basically everything, so our "B" is now "A"
//the call itself will still be in the row major assumption, were just gonna fix it under the hood


void cuda_matmul(const float* d_A, const float* d_B, float* d_C,
                 int M, int K, int N) {

    cublasHandle_t handle = cuda_backend_get_handle();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    //cublas func call:
    // args handle, transB. transA, N, M, K, &alpha, beta. ldb = N, A,, lda = L, &beta, C, ldc = N
    //the ldx stuff is just leading dims (number of cols in the original row major mat)

    CUBLAS_CHECK(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B, N,
                            d_A, K,
                            &beta,
                            d_C, N));

}


//same ideas for the transposed versions
void cuda_matmul_transA(const float* d_A, const float* d_B, float* d_C,
                        int M, int K, int N) {
    cublasHandle_t handle = cuda_backend_get_handle();
    float alpha = 1.0f;
    float beta  = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K,
                             &alpha,
                             d_B, N,
                             d_A, M,
                             &beta,
                             d_C, N));
}

void cuda_matmul_transB(const float* d_A, const float* d_B, float* d_C,
                        int M, int K, int N) {
    cublasHandle_t handle = cuda_backend_get_handle();
    float alpha = 1.0f;
    float beta  = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, K,
                             d_A, K,
                             &beta,
                             d_C, N));
}