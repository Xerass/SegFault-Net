#include "conv2d.h"
#include <math.h>
#include <time.h>


static Tensor* conv2d_forward(LayerBase* self, Tensor* input, int is_training){
    Conv2dLayer* l = (Conv2dLayer*)self;
    (void)is_training;

    //input expected is C_in, H, W per sample
    //batching is not handled internally here

    int C_in = l->C_in;
    int H = input->shape[1];
    int W = input->shape[2];
    int out_H = (H + 2 * l->pad - l->kH) / l->stride + 1;
    int out_W = (W + 2 * l->pad - l->kW) / l->stride + 1;

    l->H_in = H;
    l->W_in = W;

    //lazy alloc again
    int col_rows = C_in * l->kH * l->kW;
    int col_cols = out_H * out_W;

    if (!l->col || l->cached_HW != H * W) {
        if (l->col) tensor_free(l->col);
        if (l->output) tensor_free(l->output);
        if (l->dX_prev) tensor_free(l->dX_prev);
        if (l->dcol) tensor_free(l->dcol);

        l->col = tensor_create(2, (int[]){col_rows, col_cols});
        l->output = tensor_create(2, (int[]){l->C_out, col_cols});
        l->dX_prev = tensor_create(3, (int[]){C_in, H, W});
        l->dcol = tensor_create(2, (int[]){col_rows, col_cols});
        l->cached_HW = H * W;
    }


    //im2col
    tensor_im2col(input, l->col, C_in, H, W, l->kH, l->kW, l->stride, l->pad);

    //matmul
    tensor_matmul(l->W, l->col, l->output, l->C_out, col_rows, col_cols); 

    //bias addition, broadcast across spatial positions
    Tensor* ones = tensor_create(2, (int[]){1, col_cols});
    float* h_ones = (float*)malloc(col_cols * sizeof(float));
    for (int i = 0; i < col_cols; i++) h_ones[i] = 1.0f;
    tensor_from_cpu(ones, h_ones);
    free(h_ones);

    Tensor* b_broadcast = tensor_create(2, (int[]){l->C_out, col_cols});
    tensor_matmul(l->b, ones, b_broadcast, l->C_out, 1, col_cols);
    tensor_add(l->output, b_broadcast, l->output);
    tensor_free(b_broadcast);
    tensor_free(ones);   

    return l->output;

}

static Tensor* conv2d_backward(LayerBase* self, Tensor* dout) {
    Conv2dLayer* l = (Conv2dLayer*)self;

    int col_rows = l->C_in * l->kH * l->kW;
    int out_H = (l->H_in + 2 * l->pad - l->kH) / l->stride + 1;
    int out_W = (l->W_in + 2 * l->pad - l->kW) / l->stride + 1;
    int col_cols = out_H * out_W;

    /* dW = dout * col^T
       dout(C_out, col_cols) * col^T(col_cols, col_rows)
       = dW(C_out, col_rows) */
    tensor_matmul_transB(dout, l->col, l->dW,
                         l->C_out, col_cols, col_rows);

    /* db = sum(dout, axis=1) — sum across spatial positions */
    tensor_sum_rows(dout, l->db, l->C_out, col_cols);

    /* dcol = W^T * dout
       W^T(col_rows, C_out) * dout(C_out, col_cols)
       = dcol(col_rows, col_cols) */
    tensor_matmul_transA(l->W, dout, l->dcol,
                         col_rows, l->C_out, col_cols);

    /* col2im: scatter column gradients back to input shape
       dX_prev must be zeroed first because col2im accumulates. */
    tensor_zero(l->dX_prev);
    tensor_col2im(l->dcol, l->dX_prev,
                  l->C_in, l->H_in, l->W_in,
                  l->kH, l->kW, l->stride, l->pad);

    return l->dX_prev;
}


static void conv2d_update(LayerBase* self, float lr) {
    Conv2dLayer* l = (Conv2dLayer*)self;
    int w_size = l->C_out * l->C_in * l->kH * l->kW;

    Tensor* step = tensor_create(2, (int[]){l->W->shape[0], l->W->shape[1]});
    tensor_scale(l->dW, -lr, step);
    tensor_add(l->W, step, l->W);
    tensor_free(step);

    Tensor* step_b = tensor_create(2, (int[]){l->C_out, 1});
    tensor_scale(l->db, -lr, step_b);
    tensor_add(l->b, step_b, l->b);
    tensor_free(step_b);
}


static void conv2d_free(LayerBase* self) {
    Conv2dLayer* l = (Conv2dLayer*)self;
    tensor_free(l->W);
    tensor_free(l->b);
    tensor_free(l->dW);
    tensor_free(l->db);
    if (l->col) tensor_free(l->col);
    if (l->output) tensor_free(l->output);
    if (l->dX_prev) tensor_free(l->dX_prev);
    if (l->dcol) tensor_free(l->dcol);
    free(l);
}


LayerBase* conv2d_create(int C_in, int C_out, int kH, int kW,
                         int stride, int pad) {
    Conv2dLayer* l = (Conv2dLayer*)calloc(1, sizeof(Conv2dLayer));
    l->base.type = LAYER_CONV2D;
    l->base.forward = conv2d_forward;
    l->base.backward = conv2d_backward;
    l->base.update = conv2d_update;
    l->base.free_layer = conv2d_free;

    l->C_in = C_in;
    l->C_out = C_out;
    l->kH = kH;
    l->kW = kW;
    l->stride = stride;
    l->pad = pad;

   //W is currently stored as
   //(C_out, C_in*kH*kW), it is unrolled later 
    int filter_size = C_in * kH * kW;
    l->W = tensor_create(2, (int[]){C_out, filter_size});
    l->b = tensor_create(2, (int[]){C_out, 1});
    l->dW = tensor_create(2, (int[]){C_out, filter_size});
    l->db = tensor_create(2, (int[]){C_out, 1});

    //Kaiming Initialization for weights
    float std = sqrtf(2.0f / filter_size);
    float* h_W = (float*)malloc(C_out * filter_size * sizeof(float));
    srand((unsigned int)time(NULL));
    for (int i = 0; i < C_out * filter_size; i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        h_W[i] = std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    }
    tensor_from_cpu(l->W, h_W);
    free(h_W);

    tensor_zero(l->b);
    tensor_zero(l->dW);
    tensor_zero(l->db);

    l->col = NULL;
    l->output = NULL;
    l->dX_prev = NULL;
    l->dcol = NULL;
    l->cached_HW = 0;

    return (LayerBase*)l;
}