#ifndef TRAINER_H
#define TRAINER_H

#include "../model/model.h"
#include "../optim/optimizer.h"
#include "../loss/loss.h"
#include "dataloader.h"


//orhcestrator for the entire operation, serves as the binding layer between
//all above

typedef struct{
    LossType loss_type;
    int epochs;
    int print_every; 
} TrainConfig;



//for each opeich under fit, we shuffledata
//model_zero_grad
//model forward
//loss compute and grad
//model backward
//optim


void trainer_fit(Model* model, Optimizer* opt, DataLoader* train_dl, DataLoader* val_dl, const TrainConfig* cfg);


//run an inference over the full dataloader, no gradietns
float trainer_evaluate(Model* model, DataLoader* dl, LossType loss_type);

//single forward pass, no need for dataloader, just a tensor and a model
Tensor* trainer_predict(Model* model, Tensor* input);
#endif