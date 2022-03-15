/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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
#endif

PROFILING_ENABLE_EXTERN(repeat_nn);
PROFILING_ENABLE_EXTERN(d_repeat_nn);
PROFILING_ENABLE_EXTERN(select);
PROFILING_ENABLE_EXTERN(select_back);
PROFILING_ENABLE_EXTERN(set_select);
PROFILING_ENABLE_EXTERN(set_select_back);
PROFILING_ENABLE_EXTERN(transform);

namespace tensorNN {

    // Deprecated. Used in UpSampling2D.
    // Repeats the rows and columns of the data by size[0] and size[1] respectively.
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
        PROFILING_FOOTER(set_select_back);
    }

    void transform(Tensor *A, Tensor* B, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform) {

        PROFILING_HEADER(transform);

        if (A->isCPU() && B->isCPU()) {
#ifdef cFPGA
            fpga_transform_nn(A, B, copy_cpu_to_fpga, copy_fpga_to_cpu, transform);
#endif
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            printf("Error, transform_nn not implemented in GPU\n");
            exit(1);
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
    }


    void quantize_linear(Tensor *A, Tensor *B, Tensor *y_scale, Tensor *y_zero_point, int axis) {


	    if (A->isCPU() && B->isCPU()) {
	      cpu_quantize_linear(A, B, y_scale, y_zero_point, axis);
	    }
#ifdef cGPU
	    else if (A->isGPU() && B->isGPU()) {
          gpu_quantize_linear(A, B, y_scale, y_zero_point, axis);
	    }
#endif
    }

    void dequantize_linear(Tensor *A, Tensor *B, Tensor *x_scale, Tensor *x_zero_point, int axis) {


	    if (A->isCPU() && B->isCPU()) {
	      cpu_dequantize_linear(A, B, x_scale, x_zero_point, axis);
	    }
#ifdef cGPU
	    else if (A->isGPU() && B->isGPU()) {
          gpu_dequantize_linear(A, B, x_scale, x_zero_point, axis);
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
    }

}
