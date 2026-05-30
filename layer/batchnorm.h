#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "layer.h"

//normalization adjusts the features across the batch to a zero mean, 1 variance
//then scales and shifts from learnable gamma and beta


//Forward:
//     mean = (1/B) * sum(x)
//     var  = (1/B) * sum((x - mean)^2)
//     x_hat = (x - mean) / sqrt(var + epsilon)
//     y = gamma * x_hat + beta


typedef struct{
    LayerBase base;
    int features; //num of feats / channels / cols
    float epsilon;

    Tensor* gamma; //init 1
    Tensor* beta;//init 0

    Tensor* dgamma; //since its learnable, we need gradients for the 2
    Tensor* dbeta;

    
    Tensor* input_cache; //not owned, pointer to forward   
    Tensor* mean;           
    Tensor* var;            

    //output buffers
    Tensor* output;
    Tensor* dX_prev;
    int batch_size_cached;
}BatchNormLayer;

LayerBase* batchnorm_create(int features);

#endif