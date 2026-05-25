#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdio.h>

//completely arbitrary, realistically, we top out at 4 dims (with conv2D)
// but 6 gives us some leeway for future implementations

#define TENSOR_MAX_DIMS 6

//define the tensor

#endif

//at a high level, tensors are just N dimensions of a matrix
//in memoery though, we cant 1:1 map a 2x2, memoery is innately flat
//so we need to store the metadata of this N dimensional matrix
typedef struct{
    //actual data, stored as a 1D array
    float* data;
    //stores each dim's size
    int shape[TENSOR_MAX_DIMS];
    //rank of tensor (number of dims)
    int ndim;
    //total num of elements in the tensor, mainly for memory management
    int size;
} Tensor;

//allicates the mem based on size
Tensor* tensor_create(int ndim, const int* shape);

void tensor_free(Tensor* t);

//since were dealing with cuda, we need some method to transfer data from cpu to gpu and back
void tensor_from_cpu(Tensor* t, const float* h_data);

void tensor_to_cpu(const Tensor* t, float* h_data);

//zeroes a tensor, mainly for utility and inits
void tensor_zero(Tensor* t);

void tensor_print_shape(const Tensor* t);