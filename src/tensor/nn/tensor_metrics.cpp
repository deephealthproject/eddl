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

int accuracy(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::accuracy");
    if (!Tensor::eqsize(A, B)) msg("Incompatible dims", "Tensor::accuracy");
    if (A->ndim != 2) msg("Accuracy only over 2D Tensor (batch x probs)", "Tensor::Accuracy");

    int acc = 0;

    B->tsem->lock();

    if (A->isCPU()) {
        acc = cpu_accuracy(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_accuracy(A,B,&acc);
      }
#endif
#ifdef cFPGA
    else if(A->isFPGA()) 
      {
         //acc = fpga_accuracy(A,B);  
         Tensor *nA=new Tensor(A->getShape(),DEV_CPU);
         Tensor *nB=new Tensor(B->getShape(),DEV_CPU); 
         fpga_copy_from_fpga(A, nA->ptr);
         fpga_copy_from_fpga(B, nB->ptr);
         acc = cpu_accuracy(nA, nB);
         fpga_copy_to_fpga(nA->ptr, A);
         fpga_copy_to_fpga(nB->ptr, B);
         delete nA;
         delete nB;
      }
#endif
    B->tsem->unlock();
    return acc;

}
