/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "eddl/tensor/tensor.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

using namespace std;


float Tensor::trace(int k){
    Tensor *t = this->diag(k);  // Generate diagonal

    float sum_diag = t->sum();
    delete t;

    return sum_diag;
}