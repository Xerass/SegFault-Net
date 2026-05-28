#ifndef LAYER_H
#define LAYER_H

#include "../tensor/tensor.h"
#include "../tensor/tensor_ops.h"

//acts as a generic layer interface for the network
//every layer type gets an enum
typedef enum {
    LAYER_LINEAR,
    LAYER_CONV2D,
    LAYER_RELU,
    LAYER_SIGMOID,
    LAYER_TANH,
    LAYER_SOFTMAX,
    LAYER_DROPOUT,
    LAYER_BATCHNORM
} LayerType;



//layerBase will act as the base class for every layer, it will act as the FIRST field
//this allows us to cast any concrete layer to layerBase and call functions through the pointer
//basically a "superclass"

//we cover 4 main ops, forward, backward, update, and free

typedef struct LayerBase {
    LayerType type;

    /* forward pass. is_training controls dropout/batchnorm behavior.
       returns a tensor owned by this layer, caller must NOT free it. */
    Tensor* (*forward)(struct LayerBase* self, Tensor* input, int is_training);

    /* backward pass. dout is the gradient from the layer above.
       Returns gradient w.r.t. this layer's input owned by this layer. */
    Tensor* (*backward)(struct LayerBase* self, Tensor* dout);

    /* apply one optimizer step to learnable parameters.
       lr = learning rate. Layers with no parameters (ReLU, Dropout)
       set this to NULL,the model skips them. */
    void (*update)(struct LayerBase* self, float lr);

    /* free all GPU memory (weights, gradients, caches) and the struct. */
    void (*free_layer)(struct LayerBase* self);
} LayerBase;


#endif