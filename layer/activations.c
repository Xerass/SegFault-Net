#include "activations.h"
#include <math.h>

//helper, solely for cache chcks
static void ensure_cache(Tensor** t, int ndim, const int* shape, int* cached_size, int new_size) {
    if (!(*t) || *cached_size != new_size) {
        if (*t) tensor_free(*t);
        *t = tensor_create(ndim, shape);
        *cached_size = new_size;
    }
}

static Tensor* relu_forward(LayerBase* self, Tensor* input, int is_training) {
    ReluLayer* l = (ReluLayer*)self;
    (void)is_training;

    ensure_cache(&l->output, input->ndim, input->shape, &l->cached_size, input->size);
    
    //check if the backward cache needs to be reallocated if batch size changed
    if (!l->dX_prev || l->dX_prev->size != input->size) {
        if (l->dX_prev) tensor_free(l->dX_prev);
        l->dX_prev = tensor_create(input->ndim, input->shape);
    }

    l->Z_input = input;

    tensor_relu_forward(input, l->output);
    return l->output;
}


static Tensor* relu_backward(LayerBase* self, Tensor* dout) {
    ReluLayer* l = (ReluLayer*)self;

    //since most of the stability checks were done in forward, backwards are very straightforward
    tensor_relu_backward(l->Z_input, dout, l->dX_prev);
    return l->dX_prev;
}


static void relu_free(LayerBase* self) {
    ReluLayer* l = (ReluLayer*)self;
    if (l->output) tensor_free(l->output);
    if (l->dX_prev) tensor_free(l->dX_prev);
    free(l);
}

//standard interface with layerbase, update is null since all of our activations own no learnable param
LayerBase* relu_create(void) {
    ReluLayer* l = (ReluLayer*)calloc(1, sizeof(ReluLayer));
    l->base.type = LAYER_RELU;
    l->base.forward = relu_forward;
    l->base.backward = relu_backward;
    l->base.update = NULL; 
    l->base.free_layer = relu_free;
    return (LayerBase*)l;
}

//   Sigmoid: f(x) = 1 / (1 + exp(-x))
//  f'(x) = f(x) * (1 - f(x))
static Tensor* sigmoid_forward(LayerBase* self, Tensor* input, int is_training) {
    SigmoidLayer* l = (SigmoidLayer*)self;
    (void)is_training;

    ensure_cache(&l->output, input->ndim, input->shape, &l->cached_size, input->size);
    if (!l->dX_prev || l->dX_prev->size != input->size) {
        if (l->dX_prev) tensor_free(l->dX_prev);
        l->dX_prev = tensor_create(input->ndim, input->shape);
    }

    l->Z_input = input;
    tensor_sigmoid_forward(input, l->output);
    return l->output;
}

static Tensor* sigmoid_backward(LayerBase* self, Tensor* dout) {
    SigmoidLayer* l = (SigmoidLayer*)self;

    tensor_sigmoid_backward(l->Z_input, dout, l->dX_prev);
    return l->dX_prev;
}

static void sigmoid_free(LayerBase* self) {
    SigmoidLayer* l = (SigmoidLayer*)self;
    if (l->output) tensor_free(l->output);
    if (l->dX_prev) tensor_free(l->dX_prev);
    free(l);
}


LayerBase* sigmoid_create(void) {
    SigmoidLayer* l = (SigmoidLayer*)calloc(1, sizeof(SigmoidLayer));
    l->base.type = LAYER_SIGMOID;
    l->base.forward = sigmoid_forward;
    l->base.backward = sigmoid_backward;
    l->base.update = NULL;
    l->base.free_layer = sigmoid_free;
    return (LayerBase*)l;
}


//   Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// f'(x) = 1 - f(x)^2
static Tensor* tanh_forward(LayerBase* self, Tensor* input, int is_training) {
    TanhLayer* l = (TanhLayer*)self;
    (void)is_training;

    ensure_cache(&l->output, input->ndim, input->shape, &l->cached_size, input->size);
    if (!l->dX_prev || l->dX_prev->size != input->size) {
        if (l->dX_prev) tensor_free(l->dX_prev);
        l->dX_prev = tensor_create(input->ndim, input->shape);
    }

    l->Z_input = input;
    tensor_tanh_forward(input, l->output);
    return l->output;
}

static Tensor* tanh_backward(LayerBase* self, Tensor* dout) {
    TanhLayer* l = (TanhLayer*)self;
    tensor_tanh_backward(l->Z_input, dout, l->dX_prev);
    return l->dX_prev;
}

static void tanh_free(LayerBase* self) {
    TanhLayer* l = (TanhLayer*)self;
    if (l->output) tensor_free(l->output);
    if (l->dX_prev) tensor_free(l->dX_prev);
    free(l);
}

LayerBase* tanh_create(void) {
    TanhLayer* l = (TanhLayer*)calloc(1, sizeof(TanhLayer));
    l->base.type = LAYER_TANH;
    l->base.forward = tanh_forward;
    l->base.backward = tanh_backward;
    l->base.update = NULL;
    l->base.free_layer = tanh_free;
    return (LayerBase*)l;
}


//softmax is not elementwise
//upstream, we will force softmax to be with its canonical partner
//cross-entropy loss, so backward simplifies to output - target

static Tensor* softmax_forward(LayerBase* self, Tensor* input, int is_training) {
    SoftmaxLayer* l = (SoftmaxLayer*)self;
    (void)is_training;

    int rows = input->shape[0];
    int cols = input->shape[1];

    ensure_cache(&l->output, input->ndim, input->shape, &l->cached_size, input->size);

    //completely forgor to create a GPU version, so it lives here for now
    //shouldnt be taxing
    float* h_in = (float*)malloc(input->size * sizeof(float));
    float* h_out = (float*)malloc(input->size * sizeof(float));
    tensor_to_cpu(input, h_in);

    for (int c = 0; c < cols; c++) {
        /* Find max in this column for numerical stability */
        float max_val = h_in[0 * cols + c];
        for (int r = 1; r < rows; r++) {
            float v = h_in[r * cols + c];
            if (v > max_val) max_val = v;
        }

        /* exp(x - max) and sum */
        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            float e = expf(h_in[r * cols + c] - max_val);
            h_out[r * cols + c] = e;
            sum += e;
        }

        /* Normalize */
        for (int r = 0; r < rows; r++) {
            h_out[r * cols + c] /= sum;
        }
    }

    tensor_from_cpu(l->output, h_out);
    free(h_in);
    free(h_out);

    return l->output;
}

static Tensor* softmax_backward(LayerBase* self, Tensor* dout) {
    (void)self;
    /* when used with cross-entropy loss (the standard case),
       the loss function computes the fused gradient directly.
       This backward is a passthrough, the gradient is already
       correct by the time it reaches here. Only here in the case 
       an upgrade is needed */
    return dout;
}

static void softmax_free(LayerBase* self) {
    SoftmaxLayer* l = (SoftmaxLayer*)self;
    if (l->output) tensor_free(l->output);
    free(l);
}

LayerBase* softmax_create(void) {
    SoftmaxLayer* l = (SoftmaxLayer*)calloc(1, sizeof(SoftmaxLayer));
    l->base.type = LAYER_SOFTMAX;
    l->base.forward = softmax_forward;
    l->base.backward = softmax_backward;
    l->base.update = NULL;
    l->base.free_layer = softmax_free;
    return (LayerBase*)l;
}