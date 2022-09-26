/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/fpga/layer_hlsinf.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"     

using namespace std;

int LHLSinf::total_layers = 0;

// Constructor (one parent layer)
LHLSinf::LHLSinf(Layer * parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_bn_relu, float bn_relu_factor, int enable_add, int enable_add_relu, int upscale_factor, int dense_operation, int use_weight_buffer, int first_row_weight_buffer, 
	      int input_offset, int output_offset, string name, int dev, int mem) :
              LHLSinf(vector<Layer*> {parent}, h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr,
              enable_relu, relu_factor, enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_stm, enable_maxp, enable_avgp, enable_batch_norm, enable_bn_relu, bn_relu_factor,
              enable_add, enable_add_relu, upscale_factor, dense_operation, use_weight_buffer, first_row_weight_buffer, input_offset, output_offset, name, dev, mem) {  
};

// Constructor (multiple parent layers)
LHLSinf::LHLSinf(vector<Layer * > parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_bn_relu, float bn_relu_factor, int enable_add, int enable_add_relu, int upscale_factor, int dense_operation, int use_weight_buffer, int first_row_weight_buffer, 
	      int input_offset, int output_offset, string name, int dev, int mem)  : MLayer(name, dev, mem) {

    if(name.empty()) this->name = "HLSinf" + to_string(++total_layers);

    // HLSinf parameters
    this->H = h;   this->W = w;
    this->Ichannels = ichannels;  
    this->Ochannels = ochannels;
    this->KH = 3;  this->KW = 3;
    this->SH = sh; this->SW = sw;
    this->PT = pt; this->PB = pb;
    this->PL = pl; this->PR = pr;
    this->enable_relu       = enable_relu;
    this->relu_factor       = relu_factor;
    this->enable_clipping   = enable_clipping;
    this->min_clip          = min_clip;
    this->max_clip          = max_clip;
    this->enable_shift      = enable_shift;
    this->pos_shift         = pos_shift;
    this->dir_shift         = dir_shift;
    this->enable_stm        = enable_stm;
    this->enable_maxp       = enable_maxp;
    this->enable_avgp       = enable_avgp;
    this->enable_batch_norm = enable_batch_norm;
    this->enable_bn_relu    = enable_bn_relu;
    this->bn_relu_factor    = bn_relu_factor;
    this->enable_add        = enable_add;
    this->enable_add_relu   = enable_add_relu;
    this->upscale_factor    = upscale_factor;
    this->dense_operation   = dense_operation;
    this->use_weight_buffer = use_weight_buffer;
    this->first_row_weight_buffer = first_row_weight_buffer;
    this->weight_buffer_initialized = 0;
    this->input_offset      = input_offset;
    this->output_offset     = output_offset;

    // HLSinf supports KHxKW = 1x1 by playing with paddings and zeroing filters
    if ((kh == 1) && (kw == 1)) {
      #ifdef FPGA_DEBUG
        printf("WARNING: Adjusting HLSinf layer to support 1x1 convolutions\n");
        if ((sh != 1) && (sw != 1)) printf("WARNING: 1x1 filter adjustment with strides different from 1\n");
      #endif
      this->PT = 0; this->PB = 2; this->PL = 0; this->PR = 2;
    }

    if ((kh == 2) && (kw == 2)) {
      this->PB = this->PB + 1;
      this->PR = this->PR + 1;
#ifdef DEBUG_FPGA
      printf("WARNING, adjusting padding to support 2x2 convolutions\n");
      if ((sh != 1) && (sw != 1)) printf("WARNING: 2x2 filter adjustment with strides different from 1\n");
#endif
    }

    // The first input is the previous layer (parent), the second one is the one used for add operation
    this->input = parent[0]->output;
    if(enable_add) this->input_add = parent[1]->output;

    // We set the parents-child relationship
    for (int i = 0; i < parent.size(); ++i) {
      parent[i]->addchild(this);
      addparent(parent[i]);
    }

    // We compute the output geometry
    int HO = (H + PT + PB - KH + SH) / SH;
    int WO = (W + PL + PR - KW + SW) / SW;
    if (enable_maxp || enable_avgp) {HO = HO / 2;  WO = WO / 2;}
    if (upscale_factor == -1) {printf("Error, upscale factor in HLSinf layer constructor\n"); exit(1);}
    HO = HO * upscale_factor;
    WO = WO * upscale_factor;

    // Now, we create the tensors needed
    this->filter = new Tensor(vector<int>{ochannels, ichannels, KH, KW}, dev);

    this->bias = new Tensor(vector<int>{ochannels}, dev);
    this->batch_norm_values = new Tensor(vector<int>{ochannels*4}, dev);
    if (dense_operation) output = new Tensor({input->shape[0], ochannels * HO * WO}, dev); else output = new Tensor(vector<int>{input->shape[0], Ochannels, HO, WO}, dev);
    params.push_back(this->filter);
    params.push_back(this->bias);
    params.push_back(this->batch_norm_values);
}

void LHLSinf::allocate_output_fpga_buffer() {
  if (output->fpga_ptr == NULL) {
    if (hlsinf_output_format == HLSINF_FP32) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*sizeof(float));
    } else if (hlsinf_output_format == HLSINF_API8) {
      output->fpga_ptr = fpga_create_memory(output->size);
    } else if (hlsinf_output_format == HLSINF_APF_8_4) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
    } else if (hlsinf_output_format == HLSINF_APF_16_8) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
    } else if (hlsinf_output_format == HLSINF_APF_32_16) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*sizeof(ap_fixed<32,16>));
    } else if (hlsinf_output_format == HLSINF_APUI8) {
      output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size);
    } else {
      printf("Error (HLSinf forward), output format not supported\n");
      exit(1);
    }
  }
}

void LHLSinf::deallocate_output_fpga_buffer() {
  fpga_destroy_memory(output->fpga_ptr);
  output->fpga_ptr = NULL;
}

int LHLSinf::get_trainable_params_count() {return 0;}

void LHLSinf::resize(int batch){output->resize(batch);}

void LHLSinf::forward() {

#ifdef cFPGA
    // The first time we perform a forward operation we need to check whether buffers have been created on the FPGA device
    // If not, then we allocate them and copy the tensor into the buffer, performing data type conversion if needed
    //
    // Filters
  /*  if (filter->fpga_ptr == NULL) {
      if (hlsinf_filter_format == HLSINF_FP32) {
        // We simply create the buffer and copy the tensor into the buffer (no data type conversion needed)
        filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, filter->size * fpga_datatype_sizeof(hlsinf_filter_format));  
        fpga_copy_memory_to_fpga(filter->ptr, filter->fpga_ptr, filter->size*fpga_datatype_sizeof(hlsinf_filter_format));
      } else if (hlsinf_filter_format == HLSINF_API8) {
        // Data conversion needed (FP32->API8)
        filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, filter->size * fpga_datatype_sizeof(hlsinf_filter_format));  
        fpga_copy_memory_to_fpga_and_format(filter->ptr, filter->fpga_ptr, filter->size, HLSINF_FP32, hlsinf_filter_format);
      } else if (hlsinf_filter_format == HLSINF_APF_8_4) {
        // Data conversion needed (FP32->APF<8,4,AP_RND_ZERO,AP_SAT>)
        filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, filter->size * fpga_datatype_sizeof(hlsinf_filter_format));  
        fpga_copy_memory_to_fpga_and_format(filter->ptr, filter->fpga_ptr, filter->size, HLSINF_FP32, hlsinf_filter_format);
      } else if (hlsinf_filter_format == HLSINF_APF_16_8) {
        // Data conversion needed (FP32->APF<16,8,AP_RND_ZERO,AP_SAT>)
        filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, filter->size * fpga_datatype_sizeof(hlsinf_filter_format));  
        fpga_copy_memory_to_fpga_and_format(filter->ptr, filter->fpga_ptr, filter->size, HLSINF_FP32, hlsinf_filter_format);
      } else if (hlsinf_filter_format == HLSINF_APF_32_16) {
        // Data conversion needed (FP32->APF<16,8,AP_RND_ZERO,AP_SAT>)
        filter->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, filter->size * fpga_datatype_sizeof(hlsinf_filter_format));  
        fpga_copy_memory_to_fpga_and_format(filter->ptr, filter->fpga_ptr, filter->size, HLSINF_FP32, hlsinf_filter_format);
      } else {
        printf("Error (HLSinf forward), filter format not supported\n");
        exit(1);
      }
    }
    // Bias
    if (bias->fpga_ptr == NULL) {
      if (hlsinf_bias_format == HLSINF_FP32) {
        // No need for data conversion (FP32->FP32), we allocate the buffer and copy the bias tensor there
        bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, bias->size * fpga_datatype_sizeof(hlsinf_bias_format));  
        fpga_copy_memory_to_fpga(bias->ptr, bias->fpga_ptr, bias->size * fpga_datatype_sizeof(hlsinf_bias_format));
      } else if (hlsinf_bias_format == HLSINF_API32) {
        // Data conversion needed to API32 (FP32->API32)
        bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, bias->size * fpga_datatype_sizeof(hlsinf_bias_format));  
        fpga_copy_memory_to_fpga_and_format(bias->ptr, bias->fpga_ptr, bias->size, HLSINF_FP32, hlsinf_bias_format);
      }  else if (hlsinf_bias_format == HLSINF_API8) {
        bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, bias->size * fpga_datatype_sizeof(hlsinf_bias_format));
        fpga_copy_memory_to_fpga_and_format(bias->ptr, bias->fpga_ptr, bias->size, HLSINF_FP32, hlsinf_bias_format);
      } else if (hlsinf_bias_format == HLSINF_APF_8_4) {
        // Data conversion needed to APF_8_4 (FP32->APF<8,4,AP_RND_ZERO,AP_SAT>)
        bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, bias->size*fpga_datatype_sizeof(hlsinf_bias_format));  
        fpga_copy_memory_to_fpga_and_format(bias->ptr, bias->fpga_ptr, bias->size, HLSINF_FP32, hlsinf_bias_format);
      } else if (hlsinf_bias_format == HLSINF_APF_16_8) {
        // Data conversion needed to APF_8_4 (FP32->APF<16,8,AP_RND_ZERO,AP_SAT>)
        bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, bias->size*fpga_datatype_sizeof(hlsinf_bias_format));  
        fpga_copy_memory_to_fpga_and_format(bias->ptr, bias->fpga_ptr, bias->size, HLSINF_FP32, hlsinf_bias_format);
      } else if (hlsinf_bias_format == HLSINF_APF_32_16) {
        // Data conversion needed to APF_8_4 (FP32->APF<16,8,AP_RND_ZERO,AP_SAT>)
        bias->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, bias->size*fpga_datatype_sizeof(hlsinf_bias_format));  
        fpga_copy_memory_to_fpga_and_format(bias->ptr, bias->fpga_ptr, bias->size, HLSINF_FP32, hlsinf_bias_format);
      } else {
        printf("Error (HLSinf forward), bias format not supported\n");
        exit(1);
      }
    }
    // BatchNorm
    if (enable_batch_norm && (batch_norm_values->fpga_ptr == NULL)) {
      // BatchNorm values assumed to be always in FP32 (might not be the case!)
      batch_norm_values->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, batch_norm_values->size*sizeof(float));  
      fpga_copy_memory_to_fpga(batch_norm_values->ptr, batch_norm_values->fpga_ptr, batch_norm_values->size*sizeof(float));
    }*/
    // Output buffer, the buffer size depends on the data type
    if (output->fpga_ptr == NULL) {
      if (hlsinf_output_format == HLSINF_FP32) {
        output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*fpga_datatype_sizeof(hlsinf_output_format));
      } else if (hlsinf_output_format == HLSINF_API8) {
        output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*fpga_datatype_sizeof(hlsinf_output_format));  
      } else if (hlsinf_output_format == HLSINF_APF_8_4) {
        output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*fpga_datatype_sizeof(hlsinf_output_format));
      } else if (hlsinf_output_format == HLSINF_APF_16_8) {
        output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*fpga_datatype_sizeof(hlsinf_output_format));
      } else if (hlsinf_output_format == HLSINF_APF_32_16) {
        output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*fpga_datatype_sizeof(hlsinf_output_format));
      } else if (hlsinf_output_format == HLSINF_APUI8) {
        output->fpga_ptr = fpga_create_memory(FPGA_CLMEM_WRITE_ONLY, output->size*fpga_datatype_sizeof(hlsinf_output_format));  
      } else {
        printf("Error (HLSinf forward), output format not supported\n");
        exit(1);
      }
    }
    // Now, we call the accelerator
    fpga_hlsinf(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_batch_norm, enable_bn_relu, bn_relu_factor, enable_maxp, enable_avgp,
		            enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_add, enable_add_relu, enable_stm, upscale_factor, use_weight_buffer, first_row_weight_buffer, weight_buffer_initialized,
                this->filter, this->bias, this->batch_norm_values, this->output);
    // in case we initialized buffer we annotate it
    if (use_weight_buffer && !weight_buffer_initialized) weight_buffer_initialized = 1;
#else
    msg("LHLSinf layer only available for FPGA", "LHLSinf::forward()");
#endif
}

void LHLSinf::backward() {msg("NotImplementedError", "LHLSinf::backward");}


Layer *LHLSinf::share(int c, int bs, vector<Layer *> p) {
 auto *n = new LHLSinf(p, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor,
                      enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_stm, enable_maxp, enable_avgp,
                      enable_batch_norm, enable_bn_relu, bn_relu_factor, enable_add, enable_add_relu, upscale_factor, dense_operation, use_weight_buffer, first_row_weight_buffer, input_offset, output_offset, "HLSinf_"+to_string(c)+this->name, this->dev, this->mem_level);

 //share params and gradients
 for (int i = 0; i < n->params.size(); i++) delete n->params[i];
 n->params.clear();
 for (int i = 0; i < n->gradients.size(); i++) delete n->gradients[i];
 n->gradients.clear();
 return n;
}

Layer *LHLSinf::clone(int c, int bs, vector<Layer *> p, int todev) {
  auto *n = new LHLSinf(p, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor,
                    enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_stm, enable_maxp, enable_avgp,
                    enable_batch_norm, enable_bn_relu, bn_relu_factor, enable_add, enable_add_relu, upscale_factor, dense_operation, use_weight_buffer, first_row_weight_buffer, input_offset, output_offset, name, todev, this->mem_level);
  n->orig = this;
  return n;
}

string LHLSinf::plot(int c) {
    string s;
    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    return s;
}
