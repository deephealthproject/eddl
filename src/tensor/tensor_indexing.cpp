/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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
#endif

using namespace std;



std::pair<unsigned int*, int> Tensor::_nonzero(){
    // Non-zero indices
    if (this->isCPU()) {
        return cpu_nonzero(this);
    }
#ifdef cGPU
    else if (this->isGPU())
      {
        msg("Not yet implemented for CPU", "Tensor::_nonzero");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    return std::make_pair(nullptr, 0);
}

Tensor* Tensor::nonzero(bool sort_indices){
    // **** ONLY FOR CPU ****

    // Parse result
    std::pair<unsigned int*, int> result = this->_nonzero(); // (pointer, size)
    unsigned int* indices_ptr = result.first;
    int indices_size = result.second;

    // Cast pointer (CPU only)
    auto *new_ptr = new float[indices_size];
    for(int i=0; i<indices_size; i++){
        new_ptr[i]= static_cast<float>(indices_ptr[i]);
    }

    // (Optional) Sort indices since "_nonzero()" is a concurrent operation
    if(sort_indices){
        std::sort(new_ptr, new_ptr + indices_size);
    }

    auto* t = new Tensor({indices_size}, new_ptr, this->device); // pointer
    return t;
}


Tensor* Tensor::where(Tensor *condition, Tensor *A, Tensor *B){ // where(x > 0, x[random], y[ones])
    Tensor *t = Tensor::empty(A->getShape(), A->device);
    t->where(condition, A, B, t);
    return t;
}

void Tensor::where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::where");

    if (condition->isCPU() && A->isCPU() && B->isCPU()) {
        cpu_where(condition, A, B, C);
    }
#ifdef cGPU
    else if (condition->isGPU() && A->isGPU() && B->isGPU())
      {
        gpu_where(condition, A, B, C);
      }
#endif
#ifdef cFPGA
    else if (condition->isFPGA() && A->isFPGA() && B->isFPGA())
      {
        fpga_where(condition, A, B, C);
      }
#endif
}

void Tensor::where_back(Tensor *condition, Tensor *PD_A, Tensor *PD_B, Tensor *D){
    checkCompatibility(PD_A, PD_B, D, "Tensor::where_back");

    if (condition->isCPU() && PD_A->isCPU() && PD_B->isCPU()) {
        cpu_where_back(condition, PD_A, PD_B, D);
    }
#ifdef cGPU
    else if (condition->isGPU() && PD_A->isGPU() && PD_B->isGPU())
    {
        //gpu_where_back(condition, PD_A, PD_B, D);
    }
#endif
#ifdef cFPGA
    else if (condition->isFPGA() && A->isFPGA() && B->isFPGA())
      {
        fpga_where_back(condition, A, B, C);
      }
#endif
}