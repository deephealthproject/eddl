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


#define fpga_data_type float

enum SUBKERNELS {
  K_DATA_IN_READER,
  K_KERNEL_IN_READER,
  K_BIAS_IN_READER,
  K_INPUT_BUFFER,
  K_PADDING,
  K_CVT,
  K_MULTIPLIER,
  K_ADDER,
  K_RELU,
  K_POOL_CVT,
  K_POOL_POOLING,
  K_WRITER,
  K_SUBKERNELS
};
static const char* subkernel_names[K_SUBKERNELS] =
{
  "data_in",
  "kernel_in",
  "bias_in",
  "input_buffer",
  "padding",
  "cvt",
  "mul",
  "add",
  "relu",
  "pool_cvt",
  "pool_pooling",
  "data_out"
};

extern cl_command_queue q;

extern cl_kernel kernel_hlsinf[16][K_SUBKERNELS];

void set_callback(cl_event event, const char *queue_name);

#endif //EDDL_FPGA_INTEL_HW_H

#endif // vendor_intel
#endif
