#include "csv_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 8192

static int count_cols(const char* line){
    int count = 1;
    for (int i = 0; line[i]; i++) {
        if (line[i] == ',') count++;
    }
    return count;
}

CSV* csv_load(const char* filepath, int has_header){
    FILE* f = fopen(filepath, "r");

    if (!f){
        fprintf(stderr, "csv_load: cannot open %s\n", filepath);
        return NULL;
    }

    char line[MAX_LINE];

    //first pass, we count rows or cols
    int num_rows = 0;
    int num_cols = 0;
    int first_line = 1;

    //manual file reading
    while (fgets(line, MAX_LINE, f)){
        if (line[0] == '\n' || line[0] == '\r') continue;

        if (first_line) {
            num_cols = count_cols(line);
            first_line = 0;
            if (has_header) continue; //dont count header
        }
        num_rows++;
    }

    if (num_rows == 0 || num_cols == 0){
        fprintf(stderr, "csv_load: empty file or no data rows\n");
        fclose(f);
        return NULL;
    }


    //allocate the csv struct
    CSV* csv = (CSV*)malloc(sizeof(CSV));
    csv->num_rows = num_rows;
    csv->num_cols = num_cols;
    csv->data = (float*)malloc(num_rows * num_cols * sizeof(float));
    csv->col_names = NULL;


    //second pass,we actually go and store the data
    rewind(f);
    int row = 0;

    //parse if headr is present
    if (has_header && fgets(line, MAX_LINE, f)) {
        csv->col_names = (char**)malloc(num_cols * sizeof(char*));

        //tokenize header lines per comma
        char* token = strtok(line, ",\n\r");
        for (int c = 0; c < num_cols && token; c++) {
            csv->col_names[c] = (char*)malloc(strlen(token) + 1);
            strcpy(csv->col_names[c], token);
            token = strtok(NULL, ",\n\r");
        }
    }

    //parse every other row
    while (fgets(line, MAX_LINE, f) && row < num_rows) {
        if (line[0] == '\n' || line[0] == '\r') continue;

        char* token = strtok(line, ",\n\r");
        for (int c = 0; c < num_cols && token; c++) {
            csv->data[row * num_cols + c] = (float)atof(token);
            token = strtok(NULL, ",\n\r");
        }
        row++;
    }

    fclose(f);
    printf("Loaded CSV: %d rows x %d cols from '%s'\n",
           num_rows, num_cols, filepath);

    return csv;
}

void csv_free(CSV* csv) {
    if (!csv) return;
    free(csv->data);
    if (csv->col_names) {
        for (int i = 0; i < csv->num_cols; i++) {
            free(csv->col_names[i]);
        }
        free(csv->col_names);
    }
    free(csv);
}