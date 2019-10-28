/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

void gpu_shift_(Tensor *A, vector<int> shift, bool reshape, string mode, float constant){

}

void gpu_rotate_(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){

}

void gpu_scale_(Tensor *A, float factor, bool reshape, string mode, float constant){

}

void gpu_flip_(Tensor *A, int axis){

}

void gpu_crop_(Tensor *A, vector<int> coords_from, vector<int> coords_to){

}

void gpu_cutout_(Tensor *A, vector<int> coords_from, vector<int> coords_to){

}