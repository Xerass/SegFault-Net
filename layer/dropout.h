#ifndef DROPOUT_H
#define DROPOUT_H

#include "layer.h"

//dropout , mainly used in regularization since weights tend to go crazy
//randomly zeroes out some activations, then scales survivors by 1/(1-p)
//scaled so expected value remains the same while promoting node robustness

typedef struct {
    LayerBase base;
    float p;            
    Tensor* mask;       //binary mask of drop or no drop
    Tensor* output;
    Tensor* dX_prev;
    int cached_size;
} DropoutLayer;

LayerBase* dropout_create(float p);

#endif