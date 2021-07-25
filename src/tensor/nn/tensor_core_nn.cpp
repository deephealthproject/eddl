/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <float.h>

#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/profiling.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"

extern int next_fpga_tensor_id;
#endif

PROFILING_ENABLE_EXTERN(repeat_nn);
PROFILING_ENABLE_EXTERN(d_repeat_nn);
PROFILING_ENABLE_EXTERN(select);
PROFILING_ENABLE_EXTERN(select_back);
PROFILING_ENABLE_EXTERN(set_select);
PROFILING_ENABLE_EXTERN(set_select_back);
PROFILING_ENABLE_EXTERN(transform);

namespace tensorNN {


    void repeat_nn(Tensor *A, Tensor *B, vector<int> size) {
        if ((A->device != B->device)) msg("Tensors in different devices", "Tensor::Repeat_NN");
        if (A->ndim != B->ndim) msg("Incompatible dims", "Tensor::Repeat");

        // Check size
        for (int i = 2; i < A->ndim; i++) {
            if (A->shape[i] * size[i - 2] != B->shape[i]) {
                msg("Incompatible dimensions (size)", "Tensor::Repeat_NN");
            }
        }

        PROFILING_HEADER(repeat_nn);

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
            fpga_repeat_nn(A, B, size);
        }
#endif
        PROFILING_FOOTER(repeat_nn);
    }

    void d_repeat_nn(Tensor *D, Tensor *A, vector<int> size) {
        if ((D->device != A->device)) msg("Tensors in different devices", "Tensor::D_Repeat_NN");

        PROFILING_HEADER(d_repeat_nn);

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
            printf("d_repeat_nn not implemented in FPGA yet\n");
            exit(1);
        }
#endif
        PROFILING_FOOTER(d_repeat_nn);
    }


    void select(Tensor *A, Tensor* B, SelDescriptor *sd){

        PROFILING_HEADER(select);

        if (A->isCPU() && B->isCPU()) {
            cpu_select_nn(A, B, sd);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            gpu_select_nn(A, B, sd);
        }
#endif
#ifdef cFPGA
        else if (A->isFPGA() && B->isFPGA())
        {
            fpga_select_nn(A, B, sd);
        }
#endif
        PROFILING_FOOTER(select);
    }

    void select_back(Tensor *A, Tensor* B, SelDescriptor *sd){

        PROFILING_HEADER(select_back);

        if (A->isCPU() && B->isCPU()) {
            cpu_select_back_nn(A, B, sd);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
           gpu_select_back_nn(A, B, sd);
        }
#endif
#ifdef cFPGA
        else if (A->isFPGA() && B->isFPGA())
        {
           fpga_select_back_nn(A, B, sd);
        }
#endif
        PROFILING_FOOTER(select_back);
    }

    void set_select(Tensor *A, Tensor *B, SelDescriptor *sd){

        PROFILING_HEADER(set_select);

        if (A->isCPU() && B->isCPU()) {
            cpu_set_select_nn(A, B, sd);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            gpu_set_select_nn(A, B, sd);
        }
#endif
#ifdef cFPGA
        else if (A->isFPGA() && B->isFPGA())
        {
            fpga_set_select_nn(A, B, sd);
        }
#endif
        PROFILING_FOOTER(set_select);
    }


    void set_select_back(Tensor *A, Tensor* B, SelDescriptor *sd){

        PROFILING_HEADER(set_select_back);

        if (A->isCPU() && B->isCPU()) {
            cpu_set_select_back_nn(A, B, sd);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            gpu_set_select_back_nn(A, B, sd);
        }
#endif
#ifdef cFPGA
        else if (A->isFPGA() && B->isFPGA())
        {
            fpga_set_select_back_nn(A, B, sd);
        }
#endif
        PROFILING_FOOTER(set_select_back);
    }

    void transform(Tensor *A, Tensor* B, int mode) {

        PROFILING_HEADER(transform);

        if (A->isCPU() && B->isCPU()) {
            printf("Error, transform_nn not implemented in CPU\n");
            exit(1);
            //cpu_transform_nn(A, B, mode);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            printf("Error, transform_nn not implemented in GPU\n");
            exit(1);
//            gpu_transform_nn(A, B, mode);
        }
#endif
#ifdef cFPGA
        else if (A->isFPGA() && B->isFPGA())
        {
            fpga_transform_nn(A, B, mode);
        }
#endif
        PROFILING_FOOTER(transform);
    }


    void expand(Tensor *A, Tensor* B, ExpandDescriptor *sd){


        if (A->isCPU() && B->isCPU()) {
            cpu_expand_nn(A, B, sd);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            gpu_expand_nn(A, B, sd);
        }
#endif
#ifdef cFPGA
            else if (A->isFPGA() && B->isFPGA())
        {
//            fpga_expand_nn(A, B, sd);
        }
#endif
    }

    void expand_back(Tensor *A, Tensor* B, ExpandDescriptor *sd){


        if (A->isCPU() && B->isCPU()) {
            cpu_expand_back_nn(A, B, sd);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            gpu_expand_back_nn(A, B, sd);
        }
#endif
#ifdef cFPGA
            else if (A->isFPGA() && B->isFPGA())
        {
//           fpga_expand_back_nn(A, B, sd);
        }
#endif
    }

    void repeat_batch(Tensor *A, Tensor* B){


        if (A->isCPU() && B->isCPU()) {
            cpu_repeat_batch(A, B);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            gpu_repeat_batch(A, B);
        }
#endif
#ifdef cFPGA
        else if (A->isFPGA() && B->isFPGA())
        {
        }
#endif
    }

    void multithreshold(Tensor *A, Tensor *B, Tensor *thresholds, float out_bias, float out_scale) {


	    if (A->isCPU() && B->isCPU() && thresholds->isCPU()) {
	      cpu_multithreshold(A, B, thresholds, out_bias, out_scale);
	    }
#ifdef cGPU
	    else if (A->isGPU() && B->isGPU() && thresholds->isGPU()) {
	      printf("multithreshold not supported for GPU\n");
	      exit(1);
	    }
#endif
#ifdef cFPGA
	    else if (A->isFPGA() && B->isFPGA() && thresholds->isFPGA()) {
	      printf("multithreshold not supported yet for FPGA\n");
	      exit(1);
	    }
#endif
    }

    void topK(Tensor *A, Tensor *B, int axis, int largest, int sorted, int K) {

            if (A->isCPU() && B->isCPU()) {
              cpu_topK(A, B, axis, largest, sorted, K);
            }
#ifdef cGPU
            else if (A->isGPU() && B->isGPU()) {
              printf("topK not supported for GPU\n");
              exit(1);
            }
#endif
#ifdef cFPGA
            else if (A->isFPGA() && B->isFPGA()) {
              printf("topK not supported yet for FPGA\n");
              exit(1);
            }
#endif
    }

}
