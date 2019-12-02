/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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

#ifdef cFPGA
#include "../../hardware/fpga/tensor_hls_op.h"
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
        else if (RD->I->isFPGA())
        { 
//         msg("reduction not implemented for FPGA\n");      
           //Tensor *I; // input
           //Tensor *O; // output
           //Tensor *D; // delta
           //Tensor *ID; // parent delta
           //Tensor *S; // indexes for max,min...
           printf("FPGA::REDUCTION\n");
//           Tensor *nI=new Tensor(RD->I->getShape(),DEV_CPU);

           ReduceDescriptor *nRD=new ReduceDescriptor(RD->I,RD->axis,"sum",RD->keepdims);
           fpga_copy_from_fpga(RD->I, nRD->I->ptr);
           fpga_copy_from_fpga(RD->O, nRD->O->ptr);
           fpga_copy_from_fpga(RD->D, nRD->D->ptr); 
           fpga_copy_from_fpga(RD->ID, nRD->ID->ptr);
           fpga_copy_from_fpga(RD->S, nRD->S->ptr);  
           cpu_reduction(nRD);
           fpga_copy_to_fpga(nRD->I->ptr, RD->I); 
           fpga_copy_to_fpga(nRD->O->ptr, RD->O);
           fpga_copy_to_fpga(nRD->D->ptr, RD->D);
           fpga_copy_to_fpga(nRD->ID->ptr, RD->ID);
           fpga_copy_to_fpga(nRD->S->ptr, RD->S);
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
        msg("reduction back not implemented for FPGA\n");
      }
  #endif
}
