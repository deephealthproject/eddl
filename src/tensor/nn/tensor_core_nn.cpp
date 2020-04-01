/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "tensor/nn/tensor_nn.h"
#include "hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "hardware/gpu/gpu_tensor.h"
#include "hardware/gpu/gpu_hw.h"
#include "hardware/gpu/nn/gpu_nn.h"
#endif

// Resizing tensors
void Tensor::resize(int b, float *fptr){

    if (b==shape[0]) return;

    shape[0] = b;

    size = 1;
    for (int i = 0; i < ndim; ++i) size *= shape[i];

    int s=size;
    for(int i=0;i<ndim;i++) {
        s/=shape[i];
        stride.push_back(s);
    }

    if (isCPU()) {
        if (fptr==nullptr) {
          free(ptr);
          ptr = get_fmem(size,"Tensor::resize");
        } else {
          ptr=fptr;
        }
        if (ndim == 2) {
            ptr2=(Eigen::MatrixXf*)new Eigen::Map<Eigen::MatrixXf>(ptr, shape[1], shape[0]);
        }
    }
#ifdef cGPU
    else if (isGPU())
        {
          if (fptr==nullptr) {
            gpu_delete_tensor(gpu_device,ptr);
            ptr=gpu_create_tensor(gpu_device,size);
          }
          else {
            ptr=fptr;
          }
        }
#endif
#ifdef cFPGA
    else {
        // create FPGA Tensor
      }
#endif

}

void Tensor::resize(int b) {
  resize(b,(float *)nullptr);
}

void Tensor::resize(int b, Tensor *T) {
  resize(b,T->ptr);
}


void repeat_nn(Tensor *A, Tensor *B, vector<int> size) {
    // TODO: Should be for N dimensions, not 2 (...and generic, not just NN)

    if ((A->device != B->device)) msg("Tensors in different devices", "Tensor::Repeat_NN");
    if (A->ndim != B->ndim) msg("Incompatible dims", "Tensor::Repeat");

    // Check size
    for(int i=2; i<A->ndim; i++){
        if(A->shape[i]*size[i-2]!=B->shape[i]){
            msg("Incompatible dimensions (size)", "Tensor::Repeat_NN");
        }
    }

    if (A->isCPU() && B->isCPU()) {
        cpu_repeat_nn(A, B, size);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU()) {
        gpu_repeat_nn(A, B, size);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void d_repeat_nn(Tensor *D, Tensor *A, vector<int> size) {
    // TODO: Should be for N dimensions, not 2 (...and generic, not just NN)
    if ((D->device != A->device)) msg("Tensors in different devices", "Tensor::D_Repeat_NN");

    if (D->isCPU() && A->isCPU()) {
        cpu_d_repeat_nn(D, A, size);
    }
#ifdef cGPU
    else if (D->isGPU() && A->isGPU()) {
        gpu_d_repeat_nn(D, A, size);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}
