/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * 
 * MPI support for EDDL Library - European Distributed Deep Learning Library.
 * Version: 
 * copyright (c) 2022, Universidad Politécnica de Valencia (UPV), GAP research group
 * Date: May 2022
 * Author: GAP Research Group (UPV), contact: plopez@disca.upv.es
 * All rights reserved
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
        float* lr, int* method, int* initial_mpi_avg, int* chunks, int* use_bi8,
        int* use_distr_dataset, int* ptmodel, char* test_file,
        bool* use_cpu, int* use_mpi, int* dgt);


#endif /* UTILS_EXAMPLES_H */

