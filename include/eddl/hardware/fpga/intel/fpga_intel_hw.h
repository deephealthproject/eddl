/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA
#ifdef cFPGA_VENDOR_INTEL

#ifndef EDDL_FPGA_INTEL_HW_H
#define EDDL_FPGA_INTEL_HW_H

// --------------------------------------------------------------------------------------------------------

//OpenCL support for Intel S10MX 
#include <CL/opencl.h>
#include <CL/cl_ext_intelfpga.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
// intel opencl support functions 
#include "eddl/hardware/fpga/intel/AOCLUtils/aocl_utils.h"
// -- end of S10MX 

//#include "eddl/hardware/fpga/fpga_profile.h"
#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/descriptors/descriptors.h"


#define MAX_KERNELS 16

#define fpga_data_type float


enum SUBKERNELS {
  K_DATA_IN_READER,
  K_KERNEL_IN_READER,
  K_BIAS_IN_READER,
  K_BATCH_NORM_READER,
  K_ADD_DATA_READER,
  //K_INPUT_BUFFER,
  K_PADDING,
  K_CVT,
  K_MULTIPLIER,
  K_ADDER,
  K_RELU,
  K_POOL_CVT,
  K_POOL_POOLING,
  K_BATCH_NORM,
  K_ADD_DATA,
  K_WRITER,
  K_SUBKERNELS
};

static const char* subkernel_names[K_SUBKERNELS] =
{
  "data_in",
  "kernel_in",
  "bias_in",
  "batch_norm_in",
  "add_data_in",
  //"input_buffer",
  "padding",
  "cvt",
  "mul",
  "add",
  "relu",
  "pool_cvt",
  "pool_pooling",
  "batch_norm",
  "add_data",
  "data_out"
};

extern cl_command_queue q;

extern cl_context context;

extern cl_kernel kernel_hlsinf[MAX_KERNELS][K_SUBKERNELS];

void set_callback(cl_event event, const char *queue_name);

void fpga_device_init(int device_type = FPGA_PLATFORM_NONE);

void fpga_destroy_memory(void *ptr_fpga);
void *fpga_create_memory(unsigned long flags, long int size);
void fpga_copy_memory_to_fpga(void *ptr_cpu, void *ptr_fpga, long int size);
void fpga_copy_memory_to_fpga_and_format(void *ptr_cpu, void *ptr_fpga, long int size, int src_format, int dst_format);
void fpga_copy_memory_from_fpga(void *ptr_fpga, void *ptr_cpu, long int size);

void fpga_transform_nn(Tensor *A, Tensor *B, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform);

#endif //EDDL_FPGA_INTEL_HW_H

#endif // vendor_intel
#endif