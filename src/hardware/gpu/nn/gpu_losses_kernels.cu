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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_nn_kernels.h"
#include "../gpu_kernels.h"


__global__ void cent(float* a, float* b, float* c, long int size)
{

 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < size){
   c[thread_id_x]=0;
   if (a[thread_id_x]) c[thread_id_x]-=a[thread_id_x]*logf(b[thread_id_x]+0.00001);
   if (a[thread_id_x]!=1.0) c[thread_id_x]-=(1.0-a[thread_id_x])*logf(1.0-b[thread_id_x]+0.00001);
  }
}
