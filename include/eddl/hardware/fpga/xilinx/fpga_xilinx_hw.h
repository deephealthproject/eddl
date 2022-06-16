/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA
#ifdef cFPGA_VENDOR_XILINX

#ifndef EDDL_FPGA_XILINX_HW_H
#define EDDL_FPGA_XILINX_HW_H

// --------------------------------------------------------------------------------------------------------
//#include "eddl/hardware/fpga/fpga_profile.h"
#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/descriptors/descriptors.h"
#include <ap_fixed.h>                       // Aproximated precision fixed point support
#include <ap_int.h>                         // Aproximated precision integer support

#include "eddl/hardware/fpga/xilinx/xcl2.hpp"

#define MAX_KERNELS 16


extern cl::CommandQueue *q;

extern cl::Context *context;
extern cl::Kernel kernel_hlsinf[MAX_KERNELS];

void set_callback(cl::Event event, const char *queue_name);

void fpga_device_init(int platform_type = 0);

void *fpga_create_memory(long int size);
void *fpga_create_memory(unsigned long flags, long int size);
void fpga_copy_memory_to_fpga(void *ptr_cpu, void *ptr_fpga, long int size);
void fpga_copy_memory_to_fpga_and_format(void *ptr_cpu, void *ptr_fpga, long int size, int src_format, int dst_format);
void fpga_copy_memory_from_fpga(void *ptr_fpga, void *ptr_cpu, long int size);

void fpga_transform_nn(Tensor *A, Tensor *B, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform);

#endif //EDDL_FPGA_XILINX_HW_H

#endif // vendor_xilinx
#endif // cfpga
