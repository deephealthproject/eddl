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
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#ifndef EDDL_CPU_NN_H
#define EDDL_CPU_NN_H

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"

// Aux
float get_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize);
void add_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize,float val);

// Activations
void cpu_relu(Tensor *A, Tensor *B);
void cpu_d_relu(Tensor *D, Tensor *I, Tensor *PD);
void cpu_softmax(Tensor *A, Tensor *B);
void cpu_d_softmax(Tensor *D, Tensor *I, Tensor *PD);

// Losses
void cpu_cent(Tensor *A, Tensor *B, Tensor *C);

// Metrics
int cpu_accuracy(Tensor *A, Tensor *B);

// Conv
void cpu_conv2D(ConvolDescriptor *D);
void cpu_conv2D_grad(ConvolDescriptor *D);
void cpu_conv2D_back(ConvolDescriptor *D);

// Pool
void cpu_mpool2D(PoolDescriptor*D);
void cpu_mpool2D_back(PoolDescriptor *D);



#endif //EDDL_CPU_NN_H
