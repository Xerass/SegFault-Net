#include "optimizer.h"
#include <string.h>


//Vanilla SGD: param -= lr * grad
//with momentum: param -= lr * (momentum * prev_update + grad)

static void sgd_update_param(Tensor* param, Tensor* grad, ParamState* state,float lr, float momentum) {
    if (momentum == 0.0f) {
           //step = -lr * grad, then param += step. no momentum very simple
        Tensor* step = tensor_create(param->ndim, param->shape);
        tensor_scale(grad, -lr, step);
        tensor_add(param, step, param);
        tensor_free(step);
    } else {
        // velocity = momentum * velocity + grad
        // param -= lr * velocity
        tensor_scale(state->m, momentum, state->m);
        tensor_add(state->m, grad, state->m);

        Tensor* step = tensor_create(param->ndim, param->shape);
        tensor_scale(state->m, -lr, step);
        tensor_add(param, step, param);
        tensor_free(step);
    }
}


   
   //m = beta1 * m + (1 - beta1) * grad           (first moment: mean
   //v = beta2 * v + (1 - beta2) * grad^2         (second moment: var)
   //m_hat = m / (1 - beta1^t)                    (bias correction)
   //v_hat = v / (1 - beta2^t)                    (bias correction)
   //param -= lr * m_hat / (sqrt(v_hat) + epsilon)
   
   //bias correction compensates for m and v being initialized to zero.
   //without it, early updates would be too small because m and v
   //haven't had time to accumulate meaningful estimates.
   //sort of a rev-up period
static void adam_update_param(Tensor* param, Tensor* grad, ParamState* state,float lr, float beta1, float beta2,float epsilon, int t) {
    int size = param->size;
    //all math for adam is done here since its elementwise and not kernel level
    //TODO: optimize by doing this on GPU instead of copying to CPU, but for now this is simpler and we can optimize later if needed
    float* h_param = (float*)malloc(size * sizeof(float));
    float* h_grad  = (float*)malloc(size * sizeof(float));
    float* h_m     = (float*)malloc(size * sizeof(float));
    float* h_v     = (float*)malloc(size * sizeof(float));

    tensor_to_cpu(param, h_param);
    tensor_to_cpu(grad, h_grad);
    tensor_to_cpu(state->m, h_m);
    tensor_to_cpu(state->v, h_v);

    //bias correction denoms, weakens overtime
    float correction_m = 1.0f - powf(beta1, (float)t);
    float correction_v = 1.0f - powf(beta2, (float)t);

    for (int i = 0; i < size; i++) {
        //running mean
        h_m[i] = beta1 * h_m[i] + (1.0f - beta1) * h_grad[i];
        h_v[i] = beta2 * h_v[i] + (1.0f - beta2) * h_grad[i] * h_grad[i];

        //bias correction
        float m_hat = h_m[i] / correction_m;
        float v_hat = h_v[i] / correction_v;

        //update param
        h_param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }

    tensor_from_cpu(param, h_param);
    tensor_from_cpu(state->m, h_m);
    tensor_from_cpu(state->v, h_v);

    free(h_param);
    free(h_grad);
    free(h_m);
    free(h_v);
}


// AdaGrad: accumulates squared gradients, shrinks learning rate over time.
   
//   sum_sq += grad^2
//   param -= (lr / (sqrt(sum_sq) + epsilon)) * grad
//each param gets its own lr that decreases as gradient accumulates. good when feats are sparse

static void adagrad_update_param(Tensor* param, Tensor* grad, ParamState* state,
                                 float lr, float epsilon) {
    int size = param->size;

    //also bound to CPU
    //TODO: optimize by doing this on GPU instead of copying to CPU, but for now this is simpler and we can optimize later if needed
    float* h_param  = (float*)malloc(size * sizeof(float));
    float* h_grad   = (float*)malloc(size * sizeof(float));
    float* h_sum_sq = (float*)malloc(size * sizeof(float));

    tensor_to_cpu(param, h_param);
    tensor_to_cpu(grad, h_grad);
    tensor_to_cpu(state->v, h_sum_sq);  

    for (int i = 0; i < size; i++) {
        h_sum_sq[i] += h_grad[i] * h_grad[i];
        h_param[i] -= (lr / (sqrtf(h_sum_sq[i]) + epsilon)) * h_grad[i];
    }

    tensor_from_cpu(param, h_param);
    tensor_from_cpu(state->v, h_sum_sq);

    free(h_param);
    free(h_grad);
    free(h_sum_sq);
}


static void init_layer_states(Optimizer* opt, Model* model) {
    opt->num_layers = model->num_layers;
    opt->layer_states = (LayerOptState*)calloc(model->num_layers,
                                               sizeof(LayerOptState));

    for (int i = 0; i < model->num_layers; i++) {
        LayerBase* layer = model->layers[i];
        LayerOptState* ls = &opt->layer_states[i];

        //walks through the layers and initializes optimizer state for each learnable param. non-learnable 
        //layers just get num_params set to 0 and are skipped in updates
        //this way we can keep all the logic for updates in one place instead of having some in the layers and some in the optimizer
        switch (layer->type) {
            case LAYER_LINEAR: {
                LinearLayer* ll = (LinearLayer*)layer;
                ls->num_params = 2;

                //state for w
                ls->params[0].m = tensor_create(ll->W->ndim, ll->W->shape);
                ls->params[0].v = tensor_create(ll->W->ndim, ll->W->shape);
                tensor_zero(ls->params[0].m);
                tensor_zero(ls->params[0].v);

                //State for b
                ls->params[1].m = tensor_create(ll->b->ndim, ll->b->shape);
                ls->params[1].v = tensor_create(ll->b->ndim, ll->b->shape);
                tensor_zero(ls->params[1].m);
                tensor_zero(ls->params[1].v);
                break;
            }
            case LAYER_CONV2D: {
                Conv2dLayer* cl = (Conv2dLayer*)layer;
                ls->num_params = 2;

                ls->params[0].m = tensor_create(cl->W->ndim, cl->W->shape);
                ls->params[0].v = tensor_create(cl->W->ndim, cl->W->shape);
                tensor_zero(ls->params[0].m);
                tensor_zero(ls->params[0].v);

                ls->params[1].m = tensor_create(cl->b->ndim, cl->b->shape);
                ls->params[1].v = tensor_create(cl->b->ndim, cl->b->shape);
                tensor_zero(ls->params[1].m);
                tensor_zero(ls->params[1].v);
                break;
            }
            case LAYER_BATCHNORM: {
                BatchNormLayer* bl = (BatchNormLayer*)layer;
                ls->num_params = 2;

                ls->params[0].m = tensor_create(bl->gamma->ndim, bl->gamma->shape);
                ls->params[0].v = tensor_create(bl->gamma->ndim, bl->gamma->shape);
                tensor_zero(ls->params[0].m);
                tensor_zero(ls->params[0].v);

                ls->params[1].m = tensor_create(bl->beta->ndim, bl->beta->shape);
                ls->params[1].v = tensor_create(bl->beta->ndim, bl->beta->shape);
                tensor_zero(ls->params[1].m);
                tensor_zero(ls->params[1].v);
                break;
            }
            default:
                //non-learnable layer, no state needed
                ls->num_params = 0;
                break;
        }
    }
}


//public facing constructors
Optimizer* optimizer_create_sgd(Model* model, float lr, float momentum) {
    Optimizer* opt = (Optimizer*)calloc(1, sizeof(Optimizer));
    opt->type = OPT_SGD;
    opt->lr = lr;
    opt->momentum = momentum;

    init_layer_states(opt, model);

    return opt;
}

Optimizer* optimizer_create_adam(Model* model, float lr, float beta1, float beta2, float epsilon) {
    Optimizer* opt = (Optimizer*)calloc(1, sizeof(Optimizer));
    opt->type = OPT_ADAM;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->epsilon = epsilon;
    opt->t = 0;

    init_layer_states(opt, model);

    return opt;
}

Optimizer* optimizer_create_adagrad(Model* model, float lr, float epsilon) {
    Optimizer* opt = (Optimizer*)calloc(1, sizeof(Optimizer));
    opt->type = OPT_ADAGRAD;
    opt->lr = lr;
    opt->epsilon = epsilon;

    init_layer_states(opt, model);

    return opt;
}


void optimizer_step(Optimizer* opt, Model* model) {
    //adam needs timestamp so bias correction works properly
    if (opt->type == OPT_ADAM) {
        opt->t++;
    }

    for (int i = 0; i < model->num_layers; i++) {
        LayerBase* layer = model->layers[i];
        LayerOptState* ls = &opt->layer_states[i];

        if (ls->num_params == 0) continue; //skip non-learnable layers

        switch (layer->type) {
            case LAYER_LINEAR: {
                LinearLayer* ll = (LinearLayer*)layer;

                //update each layer  param based on optimizer type, using the same state struct but different fields for adam vs sgd
                if (opt->type == OPT_SGD) {
                    sgd_update_param(ll->W, ll->dW, &ls->params[0],
                                     opt->lr, opt->momentum);
                    sgd_update_param(ll->b, ll->db, &ls->params[1],
                                     opt->lr, opt->momentum);
                } else if (opt->type == OPT_ADAM) {
                    adam_update_param(ll->W, ll->dW, &ls->params[0],
                                     opt->lr, opt->beta1, opt->beta2,
                                     opt->epsilon, opt->t);
                    adam_update_param(ll->b, ll->db, &ls->params[1],
                                     opt->lr, opt->beta1, opt->beta2,
                                     opt->epsilon, opt->t);
                } else if (opt->type == OPT_ADAGRAD) {
                    adagrad_update_param(ll->W, ll->dW, &ls->params[0],
                                         opt->lr, opt->epsilon);
                    adagrad_update_param(ll->b, ll->db, &ls->params[1],
                                         opt->lr, opt->epsilon);
                }
                break;
            }
            case LAYER_CONV2D: {
                Conv2dLayer* cl = (Conv2dLayer*)layer;

                if (opt->type == OPT_SGD) {
                    sgd_update_param(cl->W, cl->dW, &ls->params[0],
                                     opt->lr, opt->momentum);
                    sgd_update_param(cl->b, cl->db, &ls->params[1],
                                     opt->lr, opt->momentum);
                } else if (opt->type == OPT_ADAM) {
                    adam_update_param(cl->W, cl->dW, &ls->params[0],
                                     opt->lr, opt->beta1, opt->beta2,
                                     opt->epsilon, opt->t);
                    adam_update_param(cl->b, cl->db, &ls->params[1],
                                     opt->lr, opt->beta1, opt->beta2,
                                     opt->epsilon, opt->t);
                } else if (opt->type == OPT_ADAGRAD) {
                    adagrad_update_param(cl->W, cl->dW, &ls->params[0],
                                         opt->lr, opt->epsilon);
                    adagrad_update_param(cl->b, cl->db, &ls->params[1],
                                         opt->lr, opt->epsilon);
                }
                break;
            }
            case LAYER_BATCHNORM: {
                BatchNormLayer* bl = (BatchNormLayer*)layer;

                if (opt->type == OPT_SGD) {
                    sgd_update_param(bl->gamma, bl->dgamma, &ls->params[0],
                                     opt->lr, opt->momentum);
                    sgd_update_param(bl->beta, bl->dbeta, &ls->params[1],
                                     opt->lr, opt->momentum);
                } else if (opt->type == OPT_ADAM) {
                    adam_update_param(bl->gamma, bl->dgamma, &ls->params[0],
                                     opt->lr, opt->beta1, opt->beta2,
                                     opt->epsilon, opt->t);
                    adam_update_param(bl->beta, bl->dbeta, &ls->params[1],
                                     opt->lr, opt->beta1, opt->beta2,
                                     opt->epsilon, opt->t);
                } else if (opt->type == OPT_ADAGRAD) {
                    adagrad_update_param(bl->gamma, bl->dgamma, &ls->params[0],
                                         opt->lr, opt->epsilon);
                    adagrad_update_param(bl->beta, bl->dbeta, &ls->params[1],
                                         opt->lr, opt->epsilon);
                }
                break;
            }
            default:
                break;
        }
    }
}

//only frees optimizer state, not model or layers since those are owned by the trainer

void optimizer_free(Optimizer* opt) {
    if (!opt) return;

    for (int i = 0; i < opt->num_layers; i++) {
        LayerOptState* ls = &opt->layer_states[i];
        for (int j = 0; j < ls->num_params; j++) {
            if (ls->params[j].m) tensor_free(ls->params[j].m);
            if (ls->params[j].v) tensor_free(ls->params[j].v);
        }
    }

    free(opt->layer_states);
    free(opt);
}