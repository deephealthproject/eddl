/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


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
          //gpu_reduction_back(RD);
        }
  #endif
  #ifdef cFPGA
      else {

      }
  #endif
}
