/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/


#include <stdexcept>

#include "eddl/hardware/fpga/fpga_hw.h"

extern cl::CommandQueue q;
extern cl::Kernel reduce_sum2D;


void fpga_reduce(Tensor *A, Tensor *B,string mode,int* map)
{
  _profile_fpga(_FPGA_REDUCE, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_REDUCE, 1);
}
void fpga_reduce(Tensor *A, Tensor *B,string mode,MapReduceDescriptor *MD)
{
    fpga_reduce(A,B,mode,MD->ind);
}

void fpga_reduce_op(Tensor *A, Tensor *B,string op,int* map)
{
  _profile_fpga(_FPGA_REDUCE_OP, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_REDUCE_OP, 1);
}

void fpga_reduce_op(Tensor *A, Tensor *B,string op,MapReduceDescriptor *MD)
{
  fpga_reduce_op(A,B,op,MD->ind);
}

void fpga_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
    _profile_fpga(_FPGA_REDUCE_SUM2D, 0);

    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = reduce_sum2D.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = reduce_sum2D.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = reduce_sum2D.setArg(2, A->shape[0]));
    OCL_CHECK(err, err = reduce_sum2D.setArg(3, A->shape[1]));
    OCL_CHECK(err, err = reduce_sum2D.setArg(4, axis));
    OCL_CHECK(err, err = reduce_sum2D.setArg(5, incB));

    OCL_CHECK(err, err = q.enqueueTask(reduce_sum2D, NULL, &event));
    q.finish();

    _profile_fpga(_FPGA_REDUCE_SUM2D, 1);
}

void fpga_reduction(ReduceDescriptor *RD){
    _profile_fpga(_FPGA_REDUCTION, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_REDUCTION, 1);
}

void fpga_reduction_back(ReduceDescriptor *RD){
  _profile_fpga(_FPGA_REDUCTION_BACK, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
 _profile_fpga(_FPGA_REDUCTION_BACK, 1);
}
