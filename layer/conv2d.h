#ifndef CONV2D_H
#define CONV2D_H

#include "layer.h"


//conv layer got relatively simplified vy the trick we did in the kernels
//forward now resolves to 
//im2col -> matmul -> addbias -> reshape

typedef struct{
    LayerBase base;

    int C_in, C_out; //channels
    int kH, kW; //kernel dims
    int stride, pad; //conv params
    int H_in, W_in; //input dims, set on first forward

    //kearnables
    Tensor* W;              
    Tensor* b;             
    Tensor* dW;
    Tensor* db;

    //f caches
    Tensor* col;            
    Tensor* output;      

    //b caches
    Tensor* dX_prev;        
    Tensor* dcol;

    int cached_HW;      //tracks H*W of input for backward, if it changes we need to free and reallocate caches    

}Conv2dLayer;

LayerBase* conv2d_create(int C_in, int C_out, int kH, int kW,
                         int stride, int pad);

#endif