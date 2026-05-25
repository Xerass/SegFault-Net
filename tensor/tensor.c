#include "tensor.h"
#include "../cuda/cuda_mem.h"


Tensor* tensor_create(int ndim, const int* shape){
    if (ndim < 1 || ndim > TENSOR_MAX_DIMS){
        fprintf(stderr, "tensor_create: ndim %d out of range [1, %d]\n",ndim, TENSOR_MAX_DIMS);
        exit(EXIT_FAILURE);
    }

    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t){
        fprintf(stderr, "tensor_create: malloc failed\n");
        exit(EXIT_FAILURE);
    }

    t->ndim = ndim;
    t->size = 1;

    //calc for size
    for (int i = 0; i < ndim; i++){
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    for (int i = ndim; i < TENSOR_MAX_DIMS; i++) {
        t->shape[i] = 0;
    }

    //allocate memoery on the gpu as well
    t->data = cuda_malloc_float(t->size);

    return t;
}


void tensor_free(Tensor* t) {
    if (!t) return;
    cuda_free(t->data);
    free(t);
}

void tensor_from_cpu(Tensor* t, const float* h_data) {
    cuda_copy_to_device(t->data, h_data, t->size);
}

void tensor_to_cpu(const Tensor* t, float* h_data) {
    cuda_copy_to_host(h_data, t->data, t->size);
}

void tensor_zero(Tensor* t) {
    cuda_memset_zero(t->data, t->size);
}

void tensor_print_shape(const Tensor* t) {
    printf("Tensor(");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(") [%d floats, %.2f KB]\n", t->size,
           t->size * sizeof(float) / 1024.0f);
}