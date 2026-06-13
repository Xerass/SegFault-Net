#include "trainer.h"
#include <stdio.h>
#include <time.h>


void trainer_fit(Model* model, Optimizer* opt, DataLoader* train_dl, DataLoader* val_dl, const TrainConfig* cfg){
    printf("Starting training: %d epochs, %d batches/epoch, batch_size=%d\n",cfg->epochs, train_dl->num_batches, train_dl->batch_size);

    for (int epoch = 0; epoch < cfg ->epochs; epoch++){
        //track train time per epoch
        clock_t epoch_start = clock();
        float epoch_loss = 0.0f;
        int batch_count = 0;



        //shuffle
        dataloader_shuffle(train_dl);
        dataloader_reset(train_dl);



        //batch loop
        while (dataloader_next_batch(train_dl)){
            Tensor* X = train_dl->X_batch;
            Tensor* y = train_dl->y_batch;


            //set zero grad from prev batch to avoid batch contamination
            model_zero_grad(model);

            //forward
            Tensor* output = model_forward(model, X, 1);


            //loss

            float batch_loss = loss_compute(cfg->loss_type, output, y);
            Tensor* grad = loss_grad(cfg->loss_type, output, y);
        
        
        
            //backward
            model_backward(model, grad);

            optimizer_step(opt, model);

            tensor_free(grad);

            epoch_loss += batch_loss;
            batch_count++;




            //per batch logging
            if (cfg->print_every > 0 && batch_count % cfg->print_every == 0) {
                printf("  Epoch %d | Batch %d/%d | Loss: %.6f\n",
                       epoch + 1, batch_count, train_dl->num_batches, batch_loss);
            }
        }



        float avg_loss = epoch_loss / (float)batch_count;
        double elapsed = (double)(clock() - epoch_start) / CLOCKS_PER_SEC;

        printf("Epoch %d/%d | Train Loss: %.6f | Time: %.2fs",epoch + 1, cfg->epochs, avg_loss, elapsed);


        //validation
        if (val_dl){
            float val_loss = trainer_evaluate(model, val_dl, cfg->loss_type);
            printf(" | Val Loss: %.6f", val_loss);
        }

        printf("\n");
    }

    printf("Training complete.\n");
}


float trainer_evaluate(Model *model, Dataloader* dl, LossType loss_type){
    //eval mode, iterate across batches, accumulate loss but no gradients


    dataloader_reset(dl);

    float total_loss = 0.0f;
    int batch_count = 0;


    while(dataloader_next_batch(dl)){
        Tensor* output = model_forward(model, dl->X_batch);
        total_loss += loss_compute(loss_type, output, dl->Y_batch);
        batch_count++;
    }

    return total_loss / (float)batch_count;
}

Tensor* trainer_predict(Model* model, Tensor* input){
    //literally just a forward wrapper
    return model_forward(model, input, 0)
}