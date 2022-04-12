/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   utils.h
 * Author: plopez
 *
 * Created on 8 de diciembre de 2021, 18:33
 */

#ifndef UTILS_EXAMPLES_H
#define UTILS_EXAMPLES_H

#define TIMED_EXEC(title, function, secs)  \
{ \
    if (id == 0) { \
        fprintf(stdout, "\nRunning: %s\n", title); \
        fflush(stdout); \
    } \
    high_resolution_clock::time_point e1 = high_resolution_clock::now(); \
    function ; \
    high_resolution_clock::time_point e2 = high_resolution_clock::now(); \
    duration<double> time_span = e2 - e1; \
    secs = time_span.count(); \
    if (id == 0) { \
        fprintf(stdout, "\n%s. Elapsed time: %1.4f secs\n\n", title, time_span.count()); \
        fflush(stdout); \
    } \
}

#define TIME_POINT1(var)  \
    high_resolution_clock::time_point var##1 = high_resolution_clock::now(); 

#define TIME_POINT2(var,acc) \
    high_resolution_clock::time_point var##2 = high_resolution_clock::now(); \
    duration<double> var##_span = var##2 - var##1; \
    acc += var##_span.count(); 

void process_arguments(int argc, char** argv, char* path, char* tr_images,
        char* tr_labels, char* ts_images, char* ts_labels, 
        int* epochs, int* batch_size, int* num_classes, 
        int* channels, int* width, int* height, 
        float* lr, int* initial_mpi_avg, int* chunks, int* use_bi8,
        int* use_distr_dataset, int* ptmodel, char* test_file,
        bool* use_cpu, bool* use_mpi);


#endif /* UTILS_EXAMPLES_H */

