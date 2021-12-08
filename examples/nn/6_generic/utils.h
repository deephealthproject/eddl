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

void process_arguments(int argc, char** argv, char* path, char* tr_images,
        char* tr_labels, char* ts_images, char* ts_labels, 
        int* epochs, int* batch_size, int* num_classes, 
        int* channels, int* width, int* height, 
        float* lr, int* initial_mpi_avg, int* chunks, int* use_bi8,
        int* use_distr_dataset);


#endif /* UTILS_EXAMPLES_H */

