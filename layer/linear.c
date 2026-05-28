#include "linear.h"
#include <math.h>
#include <time.h>
// implementations of the 4 main operations


//outputs a tensor
static Tensor* linear_forward(LayerBase* self, Tensor* input, int is_training){
    //cast the base layer to a linear layer
    LinearLayer* l = (LinearLayer*)self;

    //get batch size by looking at input shape at the specific dimension
    int batch_size = input -> shape[input -> ndim - 1];

    //cached input pointer
    l -> X_input = input;

    //alloc and scratch tensors on first call or realloc if batch size changes
    if (!l->output || l->batch_size_cached != batch_size) {
        if (l->output) tensor_free(l->output);
        if (l->dX_prev) tensor_free(l->dX_prev);
        if (l->ones) tensor_free(l->ones);

        l->output = tensor_create(2, (int[]){l->out_features, batch_size});
        l->dX_prev = tensor_create(2, (int[]){l->in_features, batch_size});

        //ones is a row vector of size batch_size, used for bias broadcast addition
        l->ones = tensor_create(2, (int[]){1, batch_size});

        //alloc h_ones on CPU, fill with 1s, copy to GPU, free CPU memory
        float* h_ones = (float*)malloc(batch_size * sizeof(float));
        for (int i = 0; i < batch_size; i++) h_ones[i] = 1.0f;

        tensor_from_cpu(l->ones, h_ones);
        free(h_ones);

        l->batch_size_cached = batch_size;
    }  


    //matmul for W * X portion
    tensor_matmul(l->W, input, l->output,
                  l->out_features, l->in_features, batch_size);

    
    //bias broadcasting   output += b * ones^T
    //a simpler approach involves using matmul with alpha = 1, beta = 1 in cublas post broadcast 
    Tensor* b_broadcast = tensor_create(2, (int[]){l->out_features, batch_size});
    tensor_matmul(l->b, l->ones, b_broadcast,
                  l->out_features, 1, batch_size);
    tensor_add(l->output, b_broadcast, l->output);


    tensor_free(b_broadcast);

    return l->output;
}


//linear backward is surprisingly simple, just aset of matmuls

static Tensor* linear_backward(LayerBase* self, Tensor* dout) {
    LinearLayer* l = (LinearLayer*)self;
    int batch_size = dout->shape[dout->ndim - 1];
    float inv_batch = 1.0f / (float)batch_size;

    //backward requires 3 main computations: dW, db, and dX_prev


    // dW = (1/B) * dout * X_input^T
    //dout(out_features, batch_size) * X^T(batch_size, in_features)
    //= dW(out_features, in_features)
    
    tensor_matmul_transB(dout, l->X_input, l->dW,
                         l->out_features, batch_size, l->in_features);
    tensor_scale(l->dW, inv_batch, l->dW);

    //db = (1/B) * sum(dout, axis=1)
    //sum each row of dout across the batch dimension.
    //result is (out_features, 1).
    tensor_sum_rows(dout, l->db, l->out_features, batch_size);
    tensor_scale(l->db, inv_batch, l->db);

    // dX_prev = W^T * dout
    //W^T(in_features, out_features) * dout(out_features, batch_size)
    //= dX_prev(in_features, batch_size)
    //this is the gradient that flows to the previous layer. */
    tensor_matmul_transA(l->W, dout, l->dX_prev,
                         l->in_features, l->out_features, batch_size);

    return l->dX_prev;
}

static void linear_update(LayerBase* self, float lr) {
    LinearLayer* l = (LinearLayer*)self;

    // SGD, W -= lr * dW, b -= lr * db
    // simple calc, get lr to scale, then add to gradients
    // optimizer layers allow for the "improved" version of this but it works
    
    //simply create step tensors, scale, then add
    Tensor* step = tensor_create(2, (int[]){l->out_features, l->in_features});
    tensor_scale(l->dW, -lr, step);
    tensor_add(l->W, step, l->W);
    tensor_free(step);

    Tensor* step_b = tensor_create(2, (int[]){l->out_features, 1});
    tensor_scale(l->db, -lr, step_b);
    tensor_add(l->b, step_b, l->b);
    tensor_free(step_b);
}

static void linear_free(LayerBase* self) {
    LinearLayer* l = (LinearLayer*)self;

    tensor_free(l->W);
    tensor_free(l->b);
    tensor_free(l->dW);
    tensor_free(l->db);
    if (l->output) tensor_free(l->output);
    if (l->dX_prev) tensor_free(l->dX_prev);
    if (l->ones) tensor_free(l->ones);

    free(l);
}

LayerBase* linear_create(int in_features, int out_features) {
    LinearLayer* l = (LinearLayer*)calloc(1, sizeof(LinearLayer));

    // wire up the vtable (our polymorphism functions for layerbase) these pointers wil
    //is how the model calls forward/backward without knowing concrete type
    l->base.type = LAYER_LINEAR;
    l->base.forward = linear_forward;
    l->base.backward = linear_backward;
    l->base.update = linear_update;
    l->base.free_layer = linear_free;

    l->in_features = in_features;
    l->out_features = out_features;



    //create the tensor for weights, bias, and their gradients. These live on the GPU as long as the layer is alive.
    l->W = tensor_create(2, (int[]){out_features, in_features});
    l->b = tensor_create(2, (int[]){out_features, 1});
    l->dW = tensor_create(2, (int[]){out_features, in_features});
    l->db = tensor_create(2, (int[]){out_features, 1});

    //xavier uniform initilization for weights, bias init to zero
    float limit = sqrtf(6.0f / (in_features + out_features));
    int w_size = out_features * in_features;
    float* h_W = (float*)malloc(w_size * sizeof(float));
    srand((unsigned int)time(NULL));
    for (int i = 0; i < w_size; i++) {
        h_W[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    tensor_from_cpu(l->W, h_W);
    free(h_W);

    tensor_zero(l->b);

    // buffers also zeroed
    tensor_zero(l->dW);
    tensor_zero(l->db);

    //these we can allocate later upon forward call
    l->output = NULL;
    l->dX_prev = NULL;
    l->ones = NULL;
    l->batch_size_cached = 0;

    return (LayerBase*)l;
}