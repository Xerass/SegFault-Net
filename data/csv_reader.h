#ifndef CSV_READER_H
#define CSV_READER_H


//just reads a csv file into a flat float arr

typedef struct{
    float* data;
    int num_rows;
    int num_cols;
    char** col_names;
}CSV;

//has header will take over col_names
CSV* csv_load(const char* filepath, int has_header);


void csv_free(CSV* csv);


#endif