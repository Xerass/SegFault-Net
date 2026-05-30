#include "batchnorm.h"

static Tensor* batchnorm_forward(LayerBase* self, Tensor* input, int is_training){
    BatchNormLayer* l = (BatchNormLayer*)self;
    (void)is_training; // TODO: use running mean/var for inference


    int batch_size = input->shape[input->ndim - 1];

    //batch size check agaiiiiin
    if (!l->output || l->batch_size_cached != batch_size) {
        if (l->output) tensor_free(l->output);
        if (l->dX_prev) tensor_free(l->dX_prev);

        l->output = tensor_create(input->ndim, input->shape);
        l->dX_prev = tensor_create(input->ndim, input->shape);
        l->batch_size_cached = batch_size;
    }

    l->input_cache = input;

    tensor_batchnorm_forward(input, l->output,
                             l->gamma, l->beta,
                             l->mean, l->var,
                             l->features, batch_size, l->epsilon);

    return l->output;
}

static Tensor* batchnorm_backward(LayerBase* self, Tensor* dout){
    BatchNormLayer* l = (BatchNormLayer*)self;
    int batch_size = dout->shape[dout->ndim - 1];

    tensor_batchnorm_backward(dout, l->input_cache,
                              l->mean, l->var, l->gamma,
                              l->dX_prev, l->dgamma, l->dbeta,
                              l->features, batch_size, l->epsilon);

    return l->dX_prev;
}

static void batchnorm_update(LayerBase* self, float lr) {
    BatchNormLayer* l = (BatchNormLayer*)self;

    //update step: gamma -= lr * dgamma, beta -= lr * dbeta
    Tensor* step = tensor_create(1, (int[]){l->features});

    tensor_scale(l->dgamma, -lr, step);
    tensor_add(l->gamma, step, l->gamma);

    tensor_scale(l->dbeta, -lr, step);
    tensor_add(l->beta, step, l->beta);

    tensor_free(step);
}

static void batchnorm_free(LayerBase* self) {
    BatchNormLayer* l = (BatchNormLayer*)self;
    tensor_free(l->gamma);
    tensor_free(l->beta);
    tensor_free(l->dgamma);
    tensor_free(l->dbeta);
    tensor_free(l->mean);
    tensor_free(l->var);
    if (l->output) tensor_free(l->output);
    if (l->dX_prev) tensor_free(l->dX_prev);
    free(l);
}

LayerBase* batchnorm_create(int features) {
    BatchNormLayer* l = (BatchNormLayer*)calloc(1, sizeof(BatchNormLayer));
    l->base.type = LAYER_BATCHNORM;
    l->base.forward = batchnorm_forward;
    l->base.backward = batchnorm_backward;
    l->base.update = batchnorm_update;
    l->base.free_layer = batchnorm_free;

    l->features = features;
    l->epsilon = 1e-5f;

    // gamma init to 1
    //beta init to 0
    l->gamma = tensor_create(1, (int[]){features});
    l->beta = tensor_create(1, (int[]){features});
    l->dgamma = tensor_create(1, (int[]){features});
    l->dbeta = tensor_create(1, (int[]){features});
    l->mean = tensor_create(1, (int[]){features});
    l->var = tensor_create(1, (int[]){features});

    float* h_ones = (float*)malloc(features * sizeof(float));
    for (int i = 0; i < features; i++) h_ones[i] = 1.0f;
    tensor_from_cpu(l->gamma, h_ones);
    free(h_ones);

    tensor_zero(l->beta);
    tensor_zero(l->dgamma);
    tensor_zero(l->dbeta);

    l->output = NULL;
    l->dX_prev = NULL;
    l->batch_size_cached = 0;

    return (LayerBase*)l;
}