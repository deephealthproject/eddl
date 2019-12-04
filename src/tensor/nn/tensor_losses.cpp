/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "tensor_nn.h"
#include "../../hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../hardware/gpu/gpu_tensor.h"
#include "../../hardware/gpu/gpu_hw.h"
#include "../../hardware/gpu/nn/gpu_nn.h"
#endif

#ifdef cFPGA
#include "../../hardware/fpga/tensor_hls_op.h"
#endif

// Cross-Entropy: C=-(A*log(B)+(1-A)*log_(1-B))
void cent(Tensor *A, Tensor *B, Tensor *C) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::cross-entropy");
    if ((!Tensor::eqsize(A, B)) || (!Tensor::eqsize(A, C))) msg("Incompatible dims", "Tensor::cross-entropy");

    C->tsem->lock();
    if (A->isCPU()) {
        cpu_cent(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_cent(A,B,C);
      }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
      {
         //fpga_cent(A,B,C); 
         Tensor *nA=new Tensor(A->getShape(),DEV_CPU);
         Tensor *nB=new Tensor(B->getShape(),DEV_CPU);
         Tensor *nC=new Tensor(C->getShape(),DEV_CPU);
         fpga_copy_from_fpga(A, nA->ptr);
         fpga_copy_from_fpga(B, nB->ptr);
         fpga_copy_from_fpga(C, nC->ptr);
         cpu_cent(nA,nB,nC);
         fpga_copy_to_fpga(nA->ptr, A);
         fpga_copy_to_fpga(nB->ptr, B);
         fpga_copy_to_fpga(nC->ptr, C);
         delete nA; 
         delete nB;
         delete nC; 
      }
#endif
    C->tsem->unlock();
}
