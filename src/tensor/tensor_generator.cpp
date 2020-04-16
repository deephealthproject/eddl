/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include "eddl/tensor/tensor.h"
#include "eddl/hardware/cpu/cpu_hw.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_nn.h"
#endif


using namespace std;

void Tensor::rand_uniform(float v) {
    if (isCPU()) {
        cpu_rand_uniform(this, v);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_uniform(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


void Tensor::rand_signed_uniform(float v) {
    if (isCPU()) {
        cpu_rand_signed_uniform(this, v);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_signed_uniform(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif


}


void Tensor::rand_binary(float v) {
    if (isCPU()) {
        cpu_rand_binary(this, v);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_binary(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


void Tensor::rand_normal(float m, float s, bool fast_math) {
    if (isCPU()) {
        cpu_rand_normal(this, m, s, fast_math);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_normal(this,m,s);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}
