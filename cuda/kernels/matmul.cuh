#ifndef MATMUL_CUH
#define MATMUL_CUH

//were gonna need the cublas hander
#include "../cuda_backend.h"
#include "../cuda_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

// C = A * B
// A is (M x K), B is (K x N), C is (M x N)
// All row major, all revice pointers
//internally, didnt actually implement, were gonna use cublasSgemm, 
//its literally almost impossible to beat cublas

void cuda_matmul(const float* d_A, const float* d_B, float* d_C,
                 int M, int K, int N);

//C = A^T * B
//A is (K x M) but treated as transposed → (M x K)
//B is (K x N), C is (M x N)
// we utilize this for gradient calc
//in backprop this comes in at dW = dZ * A_prev^T, 
//where dZ is (M x N) and A_prev is (M x K) but treated as transposed → (K x M)
void cuda_matmul_transA(const float* d_A, const float* d_B, float* d_C,
                        int M, int K, int N);

//same concept as above only that we transpose matrix B
//this comes in when propagating back to a previous layer's linear layer
//so its typically an activation
//dA_prev = W^T * dZ
void cuda_matmul_transB(const float* d_A, const float* d_B, float* d_C,
                        int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif