/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_nn.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

// Resizing tensors
void Tensor::resize(int b, float *fptr, cl::Buffer ffpga_ptr){

    if (b==shape[0]) return;

    shape[0] = b;

    size = 1;
    for (int i = 0; i < ndim; ++i) size *= shape[i];

    int s=size;
    stride.clear();
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
	    int ant = fpga_tensor_id;
	    fpga_tensor_id = next_fpga_tensor_id;
	    next_fpga_tensor_id++;
	    printf("FPGA (resize): new tensor id %d (ant %d)\n", fpga_tensor_id, fpga_tensor_id_ant);
          }
          else {
            ptr=fptr;
          }
        }
#endif
#ifdef cFPGA
    else if (isFPGA())
        {
          if (fptr==nullptr) {
            #ifdef FPGA_DEBUG
	    printf("FPGA: resize (removing and creating new tensor (id %d)\n", fpga_tensor_id);
            #endif
            fpga_delete_tensor(fpga_device,fpga_ptr, fpga_tensor_id, size);
            fpga_ptr=fpga_create_tensor(fpga_device,size);
	    // we also manage cpu buffers (to ease the cpu emulation flow)
	    free(ptr);
	    ptr = get_fmem(size,"Tensor::resize");

          } else {
	    printf("FPGA: resize (tensor_core_nn) with just pointer assignment (will not work on FPGA)\n");
	    exit(1);
            ptr=fptr;
          }
	  // we also manage cpu buffers for eigen, to ease the cpu emulation flow
          if (ndim == 2) {
            ptr2=(Eigen::MatrixXf*)new Eigen::Map<Eigen::MatrixXf>(ptr, shape[1], shape[0]);
          }
        }
#endif

}

void Tensor::resize(int b) {
  resize(b,(float *)nullptr, (cl::Buffer)nullptr);
}

void Tensor::resize(int b, Tensor *T) {
  resize(b, T->ptr, T->fpga_ptr);
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
    else if (A->isFPGA() && B->isFPGA()) {
        fpga_repeat_nn(A, B, size);
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
    else if (D->isFPGA() && A->isFPGA()) {
        fpga_d_repeat_nn(D, A, size);
      }
#endif
}
