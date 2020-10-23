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

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

using namespace std;

void Tensor::fill_rand_uniform_(float v) {
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
        fpga_rand_uniform(this,v);
    }
#endif

}


void Tensor::fill_rand_signed_uniform_(float v) {
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
        fpga_rand_signed_uniform(this, v);
    }
#endif


}


void Tensor::fill_rand_binary_(float v) {
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
        fpga_rand_binary(this, v);
    }
#endif

}


void Tensor::fill_rand_normal_(float m, float s, bool fast_math) {
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
        fpga_rand_normal(this, m, s, fast_math);
    }
#endif

}
