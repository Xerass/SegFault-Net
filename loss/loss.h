#ifndef LOSS_H
#define LOSS_H

#include "../tensor/tensor.h"
#include "../tensor/tensor_ops.h"

//losses, computes magnitude of error for backprop

//calculats the scalar value, then the gradients for backward pass


typedef enum {
    LOSS_MSE,           //MSE typical for regression
    LOSS_CROSS_ENTROPY, //multiclass classification
    LOSS_BCE            //binary
} LossType;

//the scalar
float loss_compute(LossType type, const Tensor* output, const Tensor* target);

//the gradient
Tensor* loss_grad(LossType type, const Tensor* output, const Tensor* target);


#endif


