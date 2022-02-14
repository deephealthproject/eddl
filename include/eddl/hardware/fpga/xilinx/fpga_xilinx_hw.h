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

extern cl::CommandQueue *q;

extern cl::Kernel kernel_hlsinf[16];

void set_callback(cl::Event event, const char *queue_name);


#endif //EDDL_FPGA_XILINX_HW_H

#endif // vendor_xilinx
#endif // cfpga
