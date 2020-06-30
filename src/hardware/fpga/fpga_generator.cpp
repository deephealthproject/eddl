/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include "eddl/random.h"
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/cpu_tensor.h"


// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_rand_uniform              = 1;
char fpga_set_cpuemu_rand_signed_uniform       = 1;
char fpga_set_cpuemu_rand_binary               = 1;
char fpga_set_cpuemu_rand_normal               = 1;

// -----------------------------------------------------------------
// rand_uniform
//
void fpga_cpuemu_rand_uniform(Tensor *A, float v) {
  cpu_rand_uniform(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_rand_uniform(Tensor * A, float v){
  _profile_fpga(_FPGA_RAND_UNIFORM, 0);
#ifndef K_ENABLED_RAND_UNIFORM
  fpga_cpuemu_rand_uniform(A, v);
#else
	printf("fpga_rand_uniform should not be running on FPGA (not efficient)\n");
	exit(1);
  // cl_int err;
  // cl::Event event;
  //
  // OCL_CHECK(err, err = kernel_rand_uniform.setArg(0, *(A->fpga_ptr)));
  // OCL_CHECK(err, err = kernel_rand_uniform.setArg(1, v));
  //
  // OCL_CHECK(err, err = q.enqueueTask(kernel_rand_uniform, NULL, &event));
  // q.finish();
#endif
  _profile_fpga(_FPGA_RAND_UNIFORM, 1);
}

// -----------------------------------------------------------------
// rand_signed_uniform
//
void fpga_cpuemu_rand_signed_uniform(Tensor *A, float v) {
  cpu_rand_signed_uniform(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_rand_signed_uniform(Tensor * A, float v){
  _profile_fpga(_FPGA_RAND_SIGNED_UNIFORM, 0);
#ifndef K_ENABLED_RAND_SIGNED_UNIFORM
  fpga_cpuemu_rand_signed_uniform(A, v);
#else
  printf("fpga_rand_signed_uniform should not be running on FPGA (not efficient)\n");
  exit(1);
  // cl_int err;
  // cl::Event event;
  //
  // OCL_CHECK(err, err = kernel_rand_signed_uniform.setArg(0, *(A->fpga_ptr)));
  // OCL_CHECK(err, err = kernel_rand_signed_uniform.setArg(1, v));
  //
  // OCL_CHECK(err, err = q.enqueueTask(kernel_rand_signed_uniform, NULL, &event));
  // q.finish();
#endif
  _profile_fpga(_FPGA_RAND_SIGNED_UNIFORM, 1);
}

// -----------------------------------------------------------------
// rand_binary
//
void fpga_cpuemu_rand_binary(Tensor *A, float v) {
  cpu_rand_binary(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_rand_binary(Tensor * A, float v){
  _profile_fpga(_FPGA_BINARY, 0);
#ifndef K_ENABLED_RAND_BINARY
  fpga_cpuemu_rand_binary(A, v);
#else
  printf("fpga_rand_binary should not be running on FPGA (not efficient)\n");
  exit(1);
  // cl_int err;
  // cl::Event event;
  //
  // OCL_CHECK(err, err = kernel_rand_binary.setArg(0, *(A->fpga_ptr)));
  // OCL_CHECK(err, err = kernel_rand_binary.setArg(1, v));
  //
  // OCL_CHECK(err, err = q.enqueueTask(kernel_rand_binary, NULL, &event));
  // q.finish();
#endif
  _profile_fpga(_FPGA_BINARY, 1);
}

// -----------------------------------------------------------------
// rand_normal
//
void fpga_cpuemu_rand_normal(Tensor *A, float m, float s, bool fast_math) {
  cpu_rand_normal(A, m, s, fast_math);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_rand_normal(Tensor * A, float m, float s, bool fast_math) {
  _profile_fpga(_FPGA_RAND_NORMAL, 0);
#ifndef K_ENABLED_RAND_NORMAL
  fpga_cpuemu_rand_normal(A, m, s, fast_math);
#else
  printf("fpga_rand_normal should not be running on FPGA (not efficient)\n");
  exit(1);
  // cl_int err;
  // cl::Event event;
  //
  // OCL_CHECK(err, err = kernel_rand_normal.setArg(0, *(A->fpga_ptr)));
  // OCL_CHECK(err, err = kernel_rand_normal.setArg(1, m));
  // OCL_CHECK(err, err = kernel_rand_normal.setArg(2, s));
  // OCL_CHECK(err, err = kernel_rand_normal.setArg(3, (bool)fast_math));
  //
  // OCL_CHECK(err, err = q.enqueueTask(kernel_rand_normal, NULL, &event));
  // q.finish();
#endif
  _profile_fpga(_FPGA_RAND_NORMAL, 0);
}

#endif
