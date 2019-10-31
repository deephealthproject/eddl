/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cmath>
#include <limits>
#include <iostream>
#include <utility>

#include "tensor.h"
#include "../hardware/cpu/cpu_hw.h"

#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#include "../hardware/gpu/gpu_hw.h"
#include "../hardware/gpu/nn/gpu_nn.h"
#endif


using namespace std;


Tensor* Tensor::shift(Tensor *A, vector<int> shift, string mode, float constant){
    if (A->isCPU()) {
        return cpu_shift_gen(A, std::move(shift), std::move(mode), constant);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        return gpu_shift(A, std::move(shift), std::move(mode), constant);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}



Tensor* Tensor::rotate(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant) {
    if (A->isCPU()) {
        return rotate(A, angle, std::move(axis), reshape, std::move(mode), constant);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        msg("Only implemented for CPU Tensors", "Tensor::rotate");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::scalef(Tensor *A, float factor, bool reshape, string mode, float constant) {
    return Tensor::scalef(A, vector<float>(A->ndim, factor), reshape, mode, constant);
}

Tensor* Tensor::scalef(Tensor *A, vector<float> factor, bool reshape, string mode, float constant){
    vector<int> new_shape(A->getShape());
    for(int i=0; i<new_shape.size(); i++){ new_shape[i] *= factor[i]; }
    return Tensor::scale(A, new_shape, reshape, std::move(mode), constant);
}

Tensor* Tensor::scale(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant) {
    if (A->isCPU()) {
        return cpu_scale(A, std::move(new_shape), reshape, std::move(mode), constant);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        return gpu_scale(A, new_shape, reshape, mode, constant);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


Tensor* Tensor::flip(Tensor *A, int axis) {
    if (A->isCPU()) {
        return cpu_flip(A, axis);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        return gpu_flip(A, axis);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::crop(Tensor *A, vector<int> coords_from, vector<int> coords_to, bool reshape, float constant) {
    if (A->isCPU()) {
        return cpu_crop(A, std::move(coords_from), std::move(coords_to), reshape, constant);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        return gpu_crop(A, std::move(coords_from), std::move(coords_to), reshape, constant);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


Tensor* Tensor::cutout(Tensor *A, vector<int> coords_from, vector<int> coords_to, float constant) {
    if (A->isCPU()) {
        return cpu_cutout(A, std::move(coords_from), std::move(coords_to), constant);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        return gpu_cutout(A, std::move(coords_from), std::move(coords_to), constant);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}
