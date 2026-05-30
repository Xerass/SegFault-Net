#include "dropout.h"
#include <time.h>


//facilitates forward pass (actual dropout logic)
static Tensor* dropout_forward(LayerBase* self, Tensor* input, int is_training){
    DropoutLayer* l = (DropoutLayer*)self;

    if (!is_training){
        return input;
    }

    //again, buffer size check
    if (!l->output || l->cached_size != input->size) {
        if (l->output) tensor_free(l->output);
        if (l->mask) tensor_free(l->mask);
        if (l->dX_prev) tensor_free(l->dX_prev);

        l->output = tensor_create(input->ndim, input->shape);
        l->mask = tensor_create(input->ndim, input->shape);
        l->dX_prev = tensor_create(input->ndim, input->shape);
        l->cached_size = input->size;
    }

    //generate a rand mask in CPU, then apply each elenent to the GPU tensors
    float scale = 1.0f / (1.0f - l->p);
    float* h_mask = (float*)malloc(input->size * sizeof(float));

    for (int i = 0; i < input->size; i++){
        float r = (float)rand() / RAND_MAX;
        //i love gambling
        h_mask[i] = (r >= l->p) ? scale : 0.0f;
    }

    tensor_from_cpu(l->mask, h_mask);
    free(h_mask);

    //apply the mask
    tensor_mul(input, l->mask, l->output);

    return l->output;
}


static Tensor* dropout_backward(LayerBase* self, Tensor* dout){
    DropoutLayer* l = (DropoutLayer*)self;

    // just reapply the same mask to the gradient
    tensor_mul(dout, l->mask, l->dX_prev);
    return l->dX_prev;
}


static void dropout_free(LayerBase* self) {
    DropoutLayer* l = (DropoutLayer*)self;
    if (l->output) tensor_free(l->output);
    if (l->mask) tensor_free(l->mask);
    if (l->dX_prev) tensor_free(l->dX_prev);
    free(l);
}

LayerBase* dropout_create(float p) {
    DropoutLayer* l = (DropoutLayer*)calloc(1, sizeof(DropoutLayer));
    l->base.type = LAYER_DROPOUT;
    l->base.forward = dropout_forward;
    l->base.backward = dropout_backward;
    l->base.update = NULL;
    l->base.free_layer = dropout_free;
    l->p = p;
    srand((unsigned int)time(NULL));
    return (LayerBase*)l;
}