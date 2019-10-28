/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_kernels.h"


// GPU: Math (in-place)
__global__ void shift_(float* a,int* shift, bool reshape, string mode, float constant){

}

__global__ void rotate_(float* a,float angle, int* axis, bool reshape, string mode, float constant){

}

__global__ void scale_(float* a,float factor, bool reshape, string mode, float constant){

}

__global__ void flip_(float* a,int axis){

}

__global__ void crop_(float* a,int* coords_from, int* coords_to){

}

__global__ void cutout_(float* a,int* coords_from, int* coords_to){

}