#ifndef MODEL_H
#define MODEL_H

#include "../layer/layer.h"

//defines sequential models, a stack of layers to be executed in order
//hoes not own any of the tensors, only the layers

//planned usage:
// Model* model = model_create(8)
// model_add(m, linear_create(d.,dasdasda)) .... you get the image

typedef struct{
    LayerBase** layers; //array of pointers to layers
    int num_layers;
    int capacity; //alloc size
}Model;

//allocates an empty model with capacity
Model* model_create(int capacity);

//appends a layer to the end of the model
void model_add(Model* m, LayerBase* layer);

//forward pass through all layers, is_training serving as a toggle, output  is final layer output, hence tensor
Tensor* model_forward(Model* m, Tensor* input, int is_training);

//backward pass through all layers in reverse, given startying loss gradient
void model_backward(Model* m, Tensor* loss_grad);

//zero all gradient buffers across all layers, best done before every training step
void model_zero_grad(Model* m);

void model_free(Model* m);

void model_summary(const Model* m);

#endif