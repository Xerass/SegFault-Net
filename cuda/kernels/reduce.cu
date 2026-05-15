#include "reduce.cuh"

#define BLOCK_SIZE 256

/*
row reduction kernel, simply sums across columns for each row
     [ a b c d ]   row 0 → sum = a+b+c+d → d_out[0]
     [ e f g h ]   row 1 → sum = e+f+g+h → d_out[1]
    [ i j k l ]   row 2 → sum = i+j+k+l → d_out[2]
     so on so fort.....

     One thread per row, each thread walks its row while accumulating the sum
     not exactly the fastest but im too dumb to do shared mem reduction yet, 
     and this is still pretty fast for the planned small batch sizes and feature counts

*/


__global__ void kernel_sum_rows(const float* in, float* out, int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int r = row; r < rows; r += stride){
        float sum = 0.0f;
        //walk every col in this row
        for(int c = 0; c < cols; c++){
            sum += in[r * cols + c];
        }
        out[r] = sum;
    }
}

//save as above but this time per col, walking over rows.
__global__ void kernel_sum_cols(const float* in, float* out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int c = col; c < cols; c += stride) {
        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            sum += in[r * cols + c];
        }
        out[c] = sum;
    }
}

static int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

void cuda_sum_rows(const float* d_in, float* d_out, int rows, int cols) {
    kernel_sum_rows<<<grid_size(rows), BLOCK_SIZE>>>(d_in, d_out, rows, cols);
}

void cuda_sum_cols(const float* d_in, float* d_out, int rows, int cols) {
    kernel_sum_cols<<<grid_size(cols), BLOCK_SIZE>>>(d_in, d_out, rows, cols);
}
