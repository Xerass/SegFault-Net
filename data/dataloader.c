#include "dataloader.h"
#include "csv_reader.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

DataLoader* dataloader_create(const float* X_data, const float* Y_data,int num_samples, int num_features,int num_targets, int batch_size){

    DataLoader* dl = (DataLoader*)calloc(1, sizeof(DataLoader));

    dl->num_samples = num_samples;
    dl->num_features = num_features;
    dl->num_targets = num_targets;
    dl->batch_size = batch_size;


    //always ceiling divide batch size
    dl->num_batches = (num_samples + batch_size - 1) / batch_size;
    dl->current_batch = 0;

    
    //copy over the data, caller should be freed from them 
    int x_size = num_samples * num_features;
    int y_size = num_samples * num_targets;
    dl->X_data = (float*)malloc(x_size * sizeof(float));
    dl->Y_data = (float*)malloc(y_size * sizeof(float));
    memcpy(dl->X_data, X_data, x_size * sizeof(float));
    memcpy(dl->Y_data, Y_data, y_size * sizeof(float));


    dl->indices = (int*)malloc(num_samples * sizeof(int));
    for (int i = 0; i < num_samples; i++){
        dl->indices[i] = i;
    }

    dl->X_batch = NULL;
    dl->Y_batch = NULL;
    dl->cached_batch_size = 0;

    srand((unsigned int)time(NULL));

    return dl;
}

DataLoader* dataloader_from_csv(const CSV* csv, int target_start_col,int num_target_cols, int batch_size){

    int num_samples = csv->num_rows;
    int num_features = csv->num_cols - num_target_cols;
    int num_targets = num_target_cols;


    //target col will be saved as target, everything else is a feature
    //copy each row, skip over the target ols
    float* X = (float*)malloc(num_samples * num_features * sizeof(float));
    float* Y = (float*)malloc(num_samples * num_targets * sizeof(float));

    for (int r = 0; r < num_samples; r++) {
        int fx = 0; /* Feature write index */
        int fy = 0; /* Target write index */

        for (int c = 0; c < csv->num_cols; c++) {
            float val = csv->data[r * csv->num_cols + c];

            if (c >= target_start_col && c < target_start_col + num_target_cols) {
                Y[r * num_targets + fy] = val;
                fy++;
            } else {
                X[r * num_features + fx] = val;
                fx++;
            }
        }
    }

    DataLoader* dl = dataloader_create(X, Y, num_samples, num_features, num_targets, batch_size);

    free(X);
    free(Y);

    return dl;
}

void dataloader_shuffle(DataLoader* dl) {
    //fisher-yates shuffle
    for (int i = dl->num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = dl->indices[i];
        dl->indices[i] = dl->indices[j];
        dl->indices[j] = tmp;
    }
}


int dataloader_next_batch(DataLoader* dl){
    //batches consumed return 0
    if (dl->current_batch >= dl->num_batches){
        return 0;
    }

    //2nd check if batch size is accurate
    int start = dl->current_batch * dl->batch_size;
    int actual_bs = dl->batch_size;
    if (start + actual_bs > dl->num_samples) {
        actual_bs = dl->num_samples - start;
    }


    //realloc GPU tensors if change occured
    if (dl->cached_batch_size != actual_bs) {
        if (dl->X_batch) tensor_free(dl->X_batch);
        if (dl->Y_batch) tensor_free(dl->Y_batch);

        dl->X_batch = tensor_create(2, (int[]){dl->num_features, actual_bs});
        dl->Y_batch = tensor_create(2, (int[]){dl->num_targets, actual_bs});
        dl->cached_batch_size = actual_bs;
    }

    //as stated earlier, we need to trainpose the data since its a features,batch setup
    float* h_X = (float*)malloc(dl->num_features * actual_bs * sizeof(float));
    float* h_Y = (float*)malloc(dl->num_targets * actual_bs * sizeof(float));

    for (int b = 0; b < actual_bs; b++) {
        int sample_idx = dl->indices[start + b];

        //transpose
        for (int f = 0; f < dl->num_features; f++) {
            h_X[f * actual_bs + b] = dl->X_data[sample_idx * dl->num_features + f];
        }
        for (int t = 0; t < dl->num_targets; t++) {
            h_Y[t * actual_bs + b] = dl->Y_data[sample_idx * dl->num_targets + t];
        }
    }

    tensor_from_cpu(dl->X_batch, h_X);
    tensor_from_cpu(dl->Y_batch, h_Y);

    free(h_X);
    free(h_Y);

    dl->current_batch++;
    return 1;

}


void dataloader_reset(DataLoader* dl) {
    dl->current_batch = 0;
}


int dataloader_current_batch_size(const DataLoader* dl) {
    int start = (dl->current_batch - 1) * dl->batch_size;
    int actual_bs = dl->batch_size;
    if (start + actual_bs > dl->num_samples) {
        actual_bs = dl->num_samples - start;
    }
    return actual_bs;
}

void dataloader_free(DataLoader* dl) {
    if (!dl) return;
    free(dl->X_data);
    free(dl->Y_data);
    free(dl->indices);
    if (dl->X_batch) tensor_free(dl->X_batch);
    if (dl->Y_batch) tensor_free(dl->Y_batch);
    free(dl);
}