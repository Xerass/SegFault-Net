#include "loss.h"
#include <math.h>

//we use log so numerical stability is a concern
#define EPS 1e-7f

//computation occurs on cpu
//should be ok since this is a one time per step thing

//MSE
//L = (1 / (N * B)) * sum((output - target)^2)
//dL/d_output = (2 / (N * B)) * (output - target)

// where : N = features, B = batch size.


static float mse_compute(const float* out, const float* tgt, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = out[i] - tgt[i];
        sum += diff * diff;
    }
    return sum / (float)size;
}

static void mse_grad_compute(const float* out, const float* tgt,
                             float* grad, int size) {
    float scale = 2.0f / (float)size;
    for (int i = 0; i < size; i++) {
        grad[i] = scale * (out[i] - tgt[i]);
    }
}


//Cross Entropy is a bit unique, since typically, we use softmax + cross_entropy
//its derivative resolves to just output - target

//L = -(1/B) * sum(target * log(output))
//dL/dZ = output - target

static float ce_compute(const float* out, const float* tgt, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        /* Clamp to prevent log(0) */
        float p = out[i] < EPS ? EPS : (out[i] > 1.0f - EPS ? 1.0f - EPS : out[i]);
        if (tgt[i] > 0.0f) {
            sum -= tgt[i] * logf(p);
        }
    }
    //divide by batch_size only, sum across feats is part of CE definition, not the averaging step
    return sum;
}

//BCE is also another special case, since its typically used with a 
//sigmoid activation, so the derivative also resolves to 
//output - target, but the loss is different than CE

//L = -(1/(N*B)) * sum(target * log(output) + (1 - target) * log(1 - output))
static float bce_compute(const float* out, const float* tgt, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float p = out[i] < EPS ? EPS : (out[i] > 1.0f - EPS ? 1.0f - EPS : out[i]);
        sum -= tgt[i] * logf(p) + (1.0f - tgt[i]) * logf(1.0f - p);
    }
    return sum / (float)size;
}

//for both CE and BCE
static void fused_grad_compute(const float* out, const float* tgt,
                               float* grad, int size, int batch_size) {
    float inv_batch = 1.0f / (float)batch_size;
    for (int i = 0; i < size; i++) {
        grad[i] = (out[i] - tgt[i]) * inv_batch;
    }
}

//public facing funcs
float loss_compute(LossType type, const Tensor* output, const Tensor* target) {
    int size = output->size;

    //move out tensors to cpu for loss computation
    float* h_out = (float*)malloc(size * sizeof(float));
    float* h_tgt = (float*)malloc(size * sizeof(float));
    tensor_to_cpu(output, h_out);
    tensor_to_cpu(target, h_tgt);

    float loss;
    switch (type) {
        case LOSS_MSE:           loss = mse_compute(h_out, h_tgt, size); break;
        case LOSS_CROSS_ENTROPY: loss = ce_compute(h_out, h_tgt, size);  break;
        case LOSS_BCE:           loss = bce_compute(h_out, h_tgt, size); break;
        default:
            fprintf(stderr, "Unknown loss type\n");
            exit(EXIT_FAILURE);
    }

    free(h_out);
    free(h_tgt);
    return loss;
}

Tensor* loss_grad(LossType type, const Tensor* output, const Tensor* target) {
    int size = output->size;

    float* h_out  = (float*)malloc(size * sizeof(float));
    float* h_tgt  = (float*)malloc(size * sizeof(float));
    float* h_grad = (float*)malloc(size * sizeof(float));
    tensor_to_cpu(output, h_out);
    tensor_to_cpu(target, h_tgt);

    switch (type) {
        case LOSS_MSE:
            mse_grad_compute(h_out, h_tgt, h_grad, size);
            break;
        case LOSS_CROSS_ENTROPY:
        case LOSS_BCE: {
            int batch_size = output->shape[output->ndim - 1];
            fused_grad_compute(h_out, h_tgt, h_grad, size, batch_size);
            break;
        }
        default:
            fprintf(stderr, "Unknown loss type\n");
            exit(EXIT_FAILURE);
    }

    //uploads the gradient to CPU, this creates a new tensor
    //caller needs to free it after use
    Tensor* grad = tensor_create(output->ndim, output->shape);
    tensor_from_cpu(grad, h_grad);

    free(h_out);
    free(h_tgt);
    free(h_grad);

    return grad;
}