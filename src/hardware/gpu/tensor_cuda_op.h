
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Juan Maroñas: jmaronas@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#ifndef _TENSOR_CUDA_OP_
#define _TENSOR_CUDA_OP_

#include <cuda.h>
#include "../../tensor/tensor.h"


void gpu_set(Tensor *A,float v);
void gpu_mult(Tensor *A,float v);
void gpu_sum(Tensor *A, float v);
void gpu_log(Tensor *A);
void gpu_log2(Tensor *A);
void gpu_log10(Tensor *A);
void gpu_logn(Tensor *A, float n);
void gpu_exp(Tensor *A);
void gpu_sqrt(Tensor *A);
void gpu_sqr(Tensor *A);
void gpu_pow(Tensor *A, float v);
void gpu_mask(Tensor *A,float v);
void gpu_total_sum(Tensor *A,float *tot);

void gpu_copy_to_gpu(float *nptr,Tensor *B);
void gpu_copy_from_gpu(Tensor *A,float *nptr);
void gpu_copy_gpu(Tensor *A,Tensor *B);
void gpu_fill(Tensor *A,int aini,int aend,Tensor *B,int bini,int bend,int inc);

void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC);
void gpu_sum2D(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC);
void gpu_sum(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC);

void gpu_el_mult(Tensor *A,Tensor *B,Tensor *C,int incC);
void gpu_el_div(Tensor *A,Tensor *B,Tensor *C,int incC);

void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void gpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);
void gpu_reduce_sum2D(Tensor *A,Tensor *B,int axis,int incB);

void gpu_rand_uniform(Tensor *A, float v);
void gpu_rand_signed_uniform(Tensor *A, float v);
void gpu_rand_gaussian(Tensor *A, float m,float s);
void gpu_rand_binary(Tensor *A, float v);

void gpu_cent(Tensor *A,Tensor *B,Tensor *C);
void gpu_accuracy(Tensor *A,Tensor *B,int *acc);


void gpu_relu(Tensor *A,Tensor *B);
void gpu_d_relu(Tensor *D,Tensor *I,Tensor *PD);

void gpu_softmax(Tensor *A,Tensor *B);
void gpu_d_softmax(Tensor *D,Tensor *I,Tensor *PD);

void gpu_conv2D(ConvolDescriptor *D);
void gpu_conv2D_grad(ConvolDescriptor *D);
void gpu_conv2D_back(ConvolDescriptor *D);

void gpu_mpool2D(PoolDescriptor *D);
void gpu_mpool2D_back(PoolDescriptor *D);

#endif
