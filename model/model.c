#include "model.h"
#include "../layer/linear.h"
#include "../layer/conv2d.h"
#include "../layer/batchnorm.h"
#include "../layer/activations.h"
#include "../layer/dropout.h"
#include <string.h>


//clean layer mapping enum to string for summary
static const char* layer_type_name(LayerType type) {
    switch (type) {
        case LAYER_LINEAR:    return "Linear";
        case LAYER_CONV2D:    return "Conv2D";
        case LAYER_RELU:      return "ReLU";
        case LAYER_SIGMOID:   return "Sigmoid";
        case LAYER_TANH:      return "Tanh";
        case LAYER_SOFTMAX:   return "Softmax";
        case LAYER_DROPOUT:   return "Dropout";
        case LAYER_BATCHNORM: return "BatchNorm";
        default:              return "Unknown";
    }
}



Model* model_create(int capacity) {
    //allocate model struct then per layer, init num_layers and capacity
    Model* m = (Model*)malloc(sizeof(Model));
    if (!m) {
        fprintf(stderr, "model_create: malloc failed\n");
        exit(EXIT_FAILURE);
    }

    m->layers = (LayerBase**)malloc(capacity * sizeof(LayerBase*));
    if (!m->layers) {
        fprintf(stderr, "model_create: malloc failed\n");
        exit(EXIT_FAILURE);
    }

    m->num_layers = 0;
    m->capacity = capacity;
    return m;
}

//very crude dynamic array addition, doubles capacity when needed
//TODO: restructure layers into tree, similar to how pyTorch autograd does it
void model_add(Model* m, LayerBase* layer) {
    if (m->num_layers >= m->capacity) {
        m->capacity *= 2;
        m->layers = (LayerBase**)realloc(m->layers,
                                         m->capacity * sizeof(LayerBase*));
        if (!m->layers) {
            fprintf(stderr, "model_add: realloc failed\n");
            exit(EXIT_FAILURE);
        }
    }
    m->layers[m->num_layers] = layer;
    m->num_layers++;
}

//simply runs all forwards in order passing output of one as input of next
Tensor* model_forward(Model* m, Tensor* input, int is_training) {
    Tensor* x = input;
    for (int i = 0; i < m->num_layers; i++) {
        x = m->layers[i]->forward(m->layers[i], x, is_training);
    }
    return x;
}

//litrally the same but backwards
void model_backward(Model* m, Tensor* loss_grad) {
    Tensor* grad = loss_grad;
    for (int i = m->num_layers - 1; i >= 0; i--) {
        grad = m->layers[i]->backward(m->layers[i], grad);
    }
}

void model_zero_grad(Model* m) {
    for (int i = 0; i < m->num_layers; i++) {
        LayerBase* layer = m->layers[i];

        switch (layer->type) {
            case LAYER_LINEAR: {
                LinearLayer* ll = (LinearLayer*)layer;
                tensor_zero(ll->dW);
                tensor_zero(ll->db);
                break;
            }
            case LAYER_CONV2D: {
                Conv2dLayer* cl = (Conv2dLayer*)layer;
                tensor_zero(cl->dW);
                tensor_zero(cl->db);
                break;
            }
            case LAYER_BATCHNORM: {
                BatchNormLayer* bl = (BatchNormLayer*)layer;
                tensor_zero(bl->dgamma);
                tensor_zero(bl->dbeta);
                break;
            }
            default:
                break;
        }
    }
}

void model_free(Model* m) {
    if (!m) return;
    for (int i = 0; i < m->num_layers; i++) {
        m->layers[i]->free_layer(m->layers[i]);
    }
    free(m->layers);
    free(m);
}


void model_summary(const Model* m) {
    printf("=== Model Summary ===\n");
    printf("%-4s %-12s\n", "#", "Layer");
    printf("=======================\n");
    for (int i = 0; i < m->num_layers; i++) {
        printf("%-4d %-12s\n", i, layer_type_name(m->layers[i]->type));
    }
    printf("=====================\n");
    printf("Total layers: %d\n", m->num_layers);
    printf("=====================\n");
}