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

#include "tensor.h"
#include "../hardware/cpu/cpu_hw.h"

#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#include "../hardware/gpu/gpu_hw.h"
#include "../hardware/gpu/nn/gpu_nn.h"
#endif


using namespace std;


void Tensor::shift_(vector<int> shift, bool reshape, string mode, float constant) {
    if (isCPU()) {
        cpu_shift_(this, shift, reshape, mode, constant);
    }
#ifdef cGPU
    else if (isGPU())
      {
        msg("Only implemented for CPU Tensors", "Tensor::shitf_");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::shift(Tensor *A, vector<int> shift, bool reshape, string mode, float constant){
    Tensor *t_new = A->clone();
    t_new->shift_(shift, reshape, mode, constant);
    return t_new;
}

void Tensor::rotate_(float angle, vector<int> axis, bool reshape, string mode, float constant) {
    if (isCPU()) {
        // TODO: Implement
    }
#ifdef cGPU
    else if (isGPU())
      {
        msg("Only implemented for CPU Tensors", "Tensor::rotate_");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::rotate(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant) {
    Tensor *t_new = A->clone();
    t_new->rotate_(angle, axis, reshape, mode, constant);
    return t_new;
}

//void Tensor::scale_(float factor, bool reshape, string mode, float constant) {
//    if (isCPU()) {
//        // TODO: Implement
//    }
//#ifdef cGPU
//    else if (isGPU())
//      {
//        msg("Only implemented for CPU Tensors", "Tensor::scale_");
//      }
//#endif
//#ifdef cFPGA
//    else {
//
//    }
//#endif
//}
Tensor* Tensor::scalef(Tensor *A, float factor, bool reshape, string mode, float constant) {
    return Tensor::scalef(A, vector<float>(A->ndim, factor), reshape, mode, constant);
}

Tensor* Tensor::scalef(Tensor *A, vector<float> factor, bool reshape, string mode, float constant){
    vector<int> new_shape(A->getShape());
    for(int i=0; i<new_shape.size(); i++){ new_shape[i] *= factor[i]; }
    return Tensor::scale(A, new_shape, reshape, mode, constant);
}

Tensor* Tensor::scale(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant) {
    if (A->isCPU()) {
        return cpu_scale(A, new_shape, reshape, mode, constant);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        msg("Only implemented for CPU Tensors", "Tensor::scale_");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::flip_(int axis) {
    if (isCPU()) {
        cpu_flip_(this, axis);
    }
#ifdef cGPU
    else if (isGPU())
      {
        msg("Only implemented for CPU Tensors", "Tensor::flip_");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::flip(Tensor *A, int axis) {
    Tensor *t_new = A->clone();
    t_new->flip_(axis);
    return t_new;
}

Tensor* Tensor::crop(Tensor *A, vector<int> coords_from, vector<int> coords_to, bool reshape, float constant) {
    if (A->isCPU()) {
        return cpu_crop(A, std::move(coords_from), std::move(coords_to), reshape, constant);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        msg("Only implemented for CPU Tensors", "Tensor::crop_");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::cutout_(vector<int> coords_from, vector<int> coords_to, float constant) {
    if (isCPU()) {
        cpu_cutout_(this, std::move(coords_from), std::move(coords_to), constant);
    }
#ifdef cGPU
    else if (isGPU())
      {
        msg("Only implemented for CPU Tensors", "Tensor::cutout_");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::cutout(Tensor *A, vector<int> coords_from, vector<int> coords_to, float constant) {
    Tensor *t_new = A->clone();
    t_new->cutout_(coords_from, coords_to, constant);
    return t_new;
}