/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_TENSOR_REDUCTION_H
#define EDDL_TENSOR_REDUCTION_H

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"

using namespace std;



void reduction(ReduceDescriptor *RD);
void reduction_back(ReduceDescriptor *RD);

int *get_reduction_map(Tensor *A, vector<int> axis);
void reduce(Tensor *A, Tensor *B,string mode,vector<int> axis,int* map=nullptr);
void reduce_mean(Tensor *A, Tensor *B,vector<int> axis,int* map=nullptr);
void reduce_variance(Tensor *A, Tensor *B,vector<int> axis,int* map=nullptr);
void reduce_max(Tensor *A, Tensor *B,vector<int> axis,int* map=nullptr);
void reduce_min(Tensor *A, Tensor *B,vector<int> axis,int* map=nullptr);

void reduce_op(Tensor *A, Tensor *B,string op,vector<int> axis,int* map=nullptr);
void reduce_sum(Tensor *A, Tensor *B,vector<int> axis,int* map=nullptr);
void reduce_diff(Tensor *A, Tensor *B,vector<int> axis,int* map=nullptr);
void reduce_mult(Tensor *A, Tensor *B,vector<int> axis,int* map=nullptr);
void reduce_div(Tensor *A, Tensor *B,vector<int> axis,int* map=nullptr);

void reduce(Tensor *A, Tensor *B,string mode,MapReduceDescriptor *MD);
void reduce_mean(Tensor *A, Tensor *B,MapReduceDescriptor *MD);
void reduce_variance(Tensor *A, Tensor *B,MapReduceDescriptor *MD);
void reduce_max(Tensor *A, Tensor *B,MapReduceDescriptor *MD);
void reduce_min(Tensor *A, Tensor *B,MapReduceDescriptor *MD);

void reduce_op(Tensor *A, Tensor *B,string op,MapReduceDescriptor *MD);
void reduce_sum(Tensor *A, Tensor *B,MapReduceDescriptor *MD);
void reduce_diff(Tensor *A, Tensor *B,MapReduceDescriptor *MD);
void reduce_mult(Tensor *A, Tensor *B,MapReduceDescriptor *MD);
void reduce_div(Tensor *A, Tensor *B,MapReduceDescriptor *MD);



#endif
