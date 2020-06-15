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
#include "eddl/hardware/cpu/cpu_hw.h"

extern cl::CommandQueue q;
extern cl::Kernel reduce_sum2D;

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_reduce          = 1;
char fpga_set_cpuemu_reduce_op       = 1;
char fpga_set_cpuemu_reduce_sum2D    = 1;
char fpga_set_cpuemu_reduction       = 1;
char fpga_set_cpuemu_reduction_back  = 1;

// -----------------------------------------------------------------
// reduce
//
void fpga_cpuemu_reduce(Tensor *A, Tensor *B, string mode, int* map) {
  // TODO: map should be mapped to FPGA
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_reduce(A, B, mode, map);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_reduce(Tensor *A, Tensor *B, string mode, int *map)
{
  _profile_fpga(_FPGA_REDUCE, 0);
  if (fpga_set_cpuemu_reduce == 1) {
      fpga_cpuemu_reduce(A, B, mode, map);
  } else {
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_reduce.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_reduce.setArg(1, *(B->fpga_ptr)));
//      OCL_CHECK(err, err = kernel_reduce.setArg(2, (int)mode));
      printf("Error, mode parameter not passed\n"); exit(1);
//      OCL_CHECK(err, err = kernel_reduce.setArg(3, (int)map));
      printf("Error, map pointer not passed\n"); exit(1);

      OCL_CHECK(err, err = q.enqueueTask(kernel_reduce, NULL, &event));
      q.finish();
  }
  _profile_fpga(_FPGA_REDUCE, 1);
}

// -----------------------------------------------------------------
// reduce
//
void fpga_reduce(Tensor *A, Tensor *B, string mode, MapReduceDescriptor *MD)
{
    fpga_reduce(A,B,mode,MD->ind);
}

// -----------------------------------------------------------------
// reduce_op
//
void fpga_cpuemu_reduce_op(Tensor *A, Tensor *B, string op, int *map) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_reduce_op(A, B, op, map);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_reduce_op(Tensor *A, Tensor *B, string op, int *map)
{
  _profile_fpga(_FPGA_REDUCE_OP, 0);
  if (fpga_set_cpuemu_reduce_op == 1) {
      fpga_cpuemu_reduce_op(A, B, op, map);
  } else {
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_reduce_op.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_reduce_op.setArg(1, *(B->fpga_ptr)));
      //OCL_CHECK(err, err = kernel_reduce_op.setArg(2, (int)op));
      //OCL_CHECK(err, err = kernel_reduce_op.setArg(3, (int)map));
      printf("Error, parameters not passed\n"); exit(1);

      OCL_CHECK(err, err = q.enqueueTask(kernel_reduce_op, NULL, &event));
      q.finish();
  }
  _profile_fpga(_FPGA_REDUCE_OP, 1);
}

// -----------------------------------------------------------------
// reduce_op
//
void fpga_reduce_op(Tensor *A, Tensor *B, string op, MapReduceDescriptor *MD)
{
  fpga_reduce_op(A,B,op,MD->ind);
}

// -----------------------------------------------------------------
// reduce_sum2D
//
void fpga_cpuemu_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_reduce_sum2D(A, B, axis, incB);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
  _profile_fpga(_FPGA_REDUCE_SUM2D, 0);
  if (fpga_set_cpuemu_reduce_sum2D == 1) {
      fpga_cpuemu_reduce_sum2D(A, B, axis, incB);
  } else {
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_reduce_sum2D.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_reduce_sum2D.setArg(1, *(B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_reduce_sum2D.setArg(2, A->shape[0]));
    OCL_CHECK(err, err = kernel_reduce_sum2D.setArg(3, A->shape[1]));
    OCL_CHECK(err, err = kernel_reduce_sum2D.setArg(4, axis));
    OCL_CHECK(err, err = kernel_reduce_sum2D.setArg(5, incB));

    OCL_CHECK(err, err = q.enqueueTask(kernel_reduce_sum2D, NULL, &event));
    q.finish();
  }
  _profile_fpga(_FPGA_REDUCE_SUM2D, 1);
}

// -----------------------------------------------------------------
// reduction
//
void fpga_cpuemu_reduction(ReduceDescriptor *RD) {
  // TODO: index should be mapped on FPGA (for the moment is at CPU)
  fpga_copy_from_fpga(RD->I, RD->I->ptr);
  cpu_reduction(RD);
  fpga_copy_to_fpga(RD->O->ptr, RD->O);
  // We ware that S tensor is not always created
  if (RD->S != nullptr) {fpga_copy_to_fpga(RD->S->ptr, RD->S);}
}

void fpga_reduction(ReduceDescriptor *RD){
  _profile_fpga(_FPGA_REDUCTION, 0);
  _profile_fpga_tensor(RD->I);
  if (fpga_set_cpuemu_reduction == 1) {
      fpga_cpuemu_reduction(RD);
  } else {
      printf("fpga_reduction not implemented yet\n"); exit(1);
      // cl_int err;
      // cl::Event event;
      //
      // OCL_CHECK(err, err = kernel_reduction.setArg(0, RD));
      //
      // OCL_CHECK(err, err = q.enqueueTask(kernel_reduction, NULL, &event));
      // q.finish();
  }
  _profile_fpga(_FPGA_REDUCTION, 1);
}

// -----------------------------------------------------------------
// reduction_back
//
void fpga_cpuemu_reduction_back(ReduceDescriptor *RD) {
  // input data: tensor RD->S, tensor RD->D, and index vector
  if (RD->S != nullptr) fpga_copy_from_fpga(RD->S, RD->S->ptr);
  fpga_copy_from_fpga(RD->D, RD->D->ptr);
  // TODO: For the moment index vector is on CPU always
  cpu_reduction_back(RD);
  // output data: tensor RD->ID
  fpga_copy_to_fpga(RD->ID->ptr, RD->ID);
}

void fpga_reduction_back(ReduceDescriptor *RD){
  _profile_fpga(_FPGA_REDUCTION_BACK, 0);
  if (fpga_set_cpuemu_reduction_back == 1) {
      fpga_cpuemu_reduction_back(RD);
  } else {
      printf("fpga_reduction_back not implemented yet\n"); exit(1);
      // cl_int err;
      // cl::Event event;
      //
      // OCL_CHECK(err, err = kernel_reduction.setArg(0, RD));
      //
      // OCL_CHECK(err, err = q.enqueueTask(kernel_reduction, NULL, &event));
      // q.finish();
  }
 _profile_fpga(_FPGA_REDUCTION_BACK, 1);
}
