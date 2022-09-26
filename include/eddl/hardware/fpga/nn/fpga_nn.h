/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/


#ifndef EDDL_FPGA_NN_H
#define EDDL_FPGA_NN_H

#include "eddl/hardware/fpga/fpga_profile.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"

// HLSinf accelerator
void fpga_hlsinf(Tensor *input, Tensor *input_add, int H, int W, int Ichannels, int Ochannels, 
                 int KH, int KW, int SH, int SW, int PT, int PB, int PL, int PR, 
                 int enable_relu, float relu_factor, int enable_batch_norm, int enable_bn_relu, float bn_relu_factor, int enable_maxp, int enable_avgp, 
                 int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_add, int enable_add_relu, int enable_stm, int enable_upscale, 
                 int use_weight_buffer, int first_row_weight_buffer, int weight_buffer_initialized, int input_offset, int output_offset,
                 Tensor *filter, Tensor *bias, Tensor* batch_norm_values, Tensor *output);

#endif //EDDL_FPGA_NN_H
