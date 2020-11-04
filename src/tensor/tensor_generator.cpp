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
#include "eddl/profiling.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

using namespace std;

PROFILING_ENABLE_EXTERN(fill_rand_uniform);
PROFILING_ENABLE_EXTERN(fill_rand_signed_uniform);
PROFILING_ENABLE_EXTERN(fill_rand_normal);
PROFILING_ENABLE_EXTERN(fill_rand_binary);

void Tensor::fill_rand_uniform_(float v) {

    PROFILING_HEADER(fill_rand_uniform);

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

    PROFILING_FOOTER(fill_rand_uniform);

}

Tensor* Tensor::fill_rand_uniform(float v){
    Tensor* t_new = Tensor::empty_like(this);
    t_new->fill_rand_uniform_(v);
    return t_new;
}

void Tensor::fill_rand_signed_uniform_(float v) {

    PROFILING_HEADER(fill_rand_signed_uniform);

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

    PROFILING_FOOTER(fill_rand_signed_uniform);
}

Tensor* Tensor::fill_rand_signed_uniform(float v){
    Tensor* t_new = Tensor::empty_like(this);
    t_new->fill_rand_signed_uniform_(v);
    return t_new;
}

void Tensor::fill_rand_normal_(float m, float s, bool fast_math) {

    PROFILING_HEADER(fill_rand_normal);

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

    PROFILING_FOOTER(fill_rand_normal);
}

Tensor* Tensor::fill_rand_normal(float m, float s, bool fast_math) {
    Tensor* t_new = Tensor::empty_like(this);
    t_new->fill_rand_normal_(m, s, fast_math);
    return t_new;
}

void Tensor::fill_rand_binary_(float v) {

    PROFILING_HEADER(fill_rand_binary);

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

    PROFILING_FOOTER(fill_rand_binary);
}

Tensor* Tensor::fill_rand_binary(float v) {
    Tensor* t_new = Tensor::empty_like(this);
    t_new->fill_rand_binary_(v);
    return t_new;
}
