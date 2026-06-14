#ifndef DATALOADER_H
#define DATALOADER_H

#include "../tensor/tensor.h"
#include "csv_reader.h"

//dataloader batches data to provide the trianing loop
//full dataset lives in the CPU
//btaches get sent to the GPU upon every next_batch
//we do need to transpose X, Y since raw_csv is row_,ajjor whjile cuda 
//expects a features, batch setup


typedef struct{
    float* X_data;
    float* Y_data;

    int num_samples;
    int num_features;
    int num_targets;

    int batch_size;
    int num_batches;
    int current_batch; //0 indexed

    int* indices;

    //GPU tensors for the current batch, reused and abused

    Tensor* X_batch;
    Tensor* Y_batch;


    int cached_batch_size;
}DataLoader;


//Create a data loader from raw float arrays
DataLoader* dataloader_create(const float* X_data, const float* Y_data,int num_samples, int num_features,int num_targets, int batch_size);


//ease of use add, dataloader_from_csv, just reads the CSV itself
//add a set_target col method

DataLoader* dataloader_from_csv(const CSV* csv, int target_start_col, int num_target_cols, int batch_size);


void dataloader_shuffle(DataLoader* dl);

//technically boolean
int dataloader_next_batch(DataLoader* dl);

void dataloader_reset(DataLoader* dl);

int dataloader_current_batch_size(const DataLoader* dl);

void dataloader_free(DataLoader* dl);

#endif