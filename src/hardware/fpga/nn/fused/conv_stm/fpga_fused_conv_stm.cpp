/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/fpga/fpga_hw.h"   // for buffer copies
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"  // for cpu emulation purposes
#include "eddl/profiling.h"


// -----------------------------------------------------------------
// Conv2D + Softplus + tanh + mult
//
void fpga_conv_stm(ConvolDescriptor *D)
{
    // debug and profiling
  _debug_fpga_funcs("conv_stm");
  _profile_fpga(_FPGA_CONV2D_STM, 0);
  _profile_fpga_tensor(D->I);
  _profile_fpga_tensor(D->K);
  _profile_fpga_tensor(D->bias);

  fpga_conv_stm_transform(D);

  // profiling
  _profile_fpga_tensor(D->O);
  _profile_fpga_tensor_print(D->O);
}

#endif