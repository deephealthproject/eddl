/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "tensor_reduction.h"
#include "../hardware/cpu/cpu_hw.h"


#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#include "../hardware/gpu/gpu_hw.h"
#include "../hardware/gpu/nn/gpu_nn.h"
#endif


using namespace std;


void reduction(ReduceDescriptor *RD){

    if (RD->I->isCPU()) {
      cpu_reduction(RD);
    }
    #ifdef cGPU
    else if (RD->I->isGPU())
      {
        gpu_reduction(RD);
      }
    #endif
    #ifdef cFPGA
        else {

        }
    #endif
}


void reduction_back(ReduceDescriptor *RD)
{

  if (RD->I->isCPU()) {
    cpu_reduction_back(RD);
  }
  #ifdef cGPU
  else if (RD->I->isGPU())
    {
      gpu_reduction_back(RD);
    }
  #endif
  #ifdef cFPGA
      else {

      }
  #endif
}
