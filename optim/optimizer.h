#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../model/model.h"
#include "../layer/linear.h"
#include "../layer/conv2d.h"
#include "../layer/batchnorm.h"



//updates model params with their specific gradients

//each type will store its own hyperparams and states vars

//will serve as the one actually called by the trainer, though the calculations are done
//by the layers themselves


typedef enum {
    OPT_SGD,
    OPT_ADAM,
    OPT_ADAGRAD
} OptimizerType;

//parameter states for adam, m and v AdaGrad will track one accumulaotr per param

typedef struct {
    Tensor* m;      
    Tensor* v;      
} ParamState;

typedef struct {
    ParamState params[4];   //defines learnable param count 4 is overkill but better safe than sorry
    int num_params;         //num actually used
} LayerOptState;


typedef struct {
    OptimizerType type;

    float lr;           

    //sg-speciic
    float momentum;       

    //adam specific
    float beta1;            //default at 0.9
    float beta2;           //default at 0.999
    float epsilon;         //default at 1e-8
    int t;                  //timestep counter for bias correction

    //per layer optimizer state
    LayerOptState* layer_states;
    
    int num_layers;         
} Optimizer;


Optimizer* optimizer_create_sgd(Model* model, float lr, float momentum);

Optimizer* optimizer_create_adam(Model* model, float lr, float beta1, float beta2, float epsilon);

//adapts learning based on historical gradient magnitude, trying it out
Optimizer* optimizer_create_adagrad(Model* model, float lr, float epsilon);

//general step function, will call specific update based on type
void optimizer_step(Optimizer* opt, Model* model);

void optimizer_free(Optimizer* opt);

#endif