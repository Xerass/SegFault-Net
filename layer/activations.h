#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "layer.h"

//tensor layers per activation, with their own forward and backward
//so layer flow would go like Linear(784,128) → ReLU → Linear(128,10) → Softmax


typedef struct {
    LayerBase base;
    Tensor* Z_input;    //pre activationm, cached for backward, not owned, pointer to prev layer onlu
    Tensor* output;    
    Tensor* dX_prev;   
    int cached_size;   
} ReluLayer;

LayerBase* relu_create(void);

typedef struct {
    LayerBase base;
    Tensor* Z_input;
    Tensor* output;
    Tensor* dX_prev;
    int cached_size;
} SigmoidLayer;

LayerBase* sigmoid_create(void);

typedef struct {
    LayerBase base;
    Tensor* Z_input;
    Tensor* output;
    Tensor* dX_prev;
    int cached_size;
} TanhLayer;

LayerBase* tanh_create(void);


//softmax is a little unique, since its not element wise
//softmax(z_i) = exp(z_i) / sum(exp(z_j)), it dependds on evey input in the sample

typedef struct {
    LayerBase base;
    Tensor* output;
    int cached_size;
} SoftmaxLayer;

LayerBase* softmax_create(void);

#endif