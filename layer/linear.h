#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"
//the most basic layer, major workhorse for most tasks

//Forward is A = W * X + b
//W is (out_features, in_features)
//X is (in_features, batch_size)
//b is (out_features, 1) broadcast across batch
//A is (out_features, batch_size)

//Backward computes three things
//dW = (1/B) * dZ * X^T     — weight gradient
//db      = (1/B) * sum(dZ, axis=1) — bias gradient
//dX_prev = W^T * dZ              — gradient to pass backward

//w and b are the only learnable parameters, so we need to store their gradients

typedef struct {
    LayerBase base;         //again, always first arg
    //dims
    int in_features;       
    int out_features;       
    //learnable params, lives in the gpu so long as the layer is alive
    Tensor* W;             
    Tensor* b;  

    //gradients, also live in gpu
    Tensor* dW;
    Tensor* db;

    //forward pass cache
    //Not owned, just a pointer to prev layers output
    Tensor* X_input;        

    //allocated once cache for gradients
    Tensor* dX_prev;     //gradient of prev layer   
    Tensor* output;       //result of forward

    //scratch tensor for bias broadcast addition 
    Tensor* ones;           
    int batch_size_cached; 
} LinearLayer;

/* create a linear layer. weights initialized with Xavier uniform,
   biases initialized to zero. */
LayerBase* linear_create(int in_features, int out_features);

#endif