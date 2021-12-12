/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
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
              int enable_avgp, int enable_batch_norm, int enable_add, int enable_upscale, int dense_operation, string name, int dev, int mem) :
              LHLSinf(vector<Layer*> {parent}, h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr,
              enable_relu, relu_factor, enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_stm, enable_maxp, enable_avgp, enable_batch_norm,
              enable_add, enable_upscale, dense_operation, name, dev, mem) {  
};

// Constructor (multiple parent layers)
LHLSinf::LHLSinf(vector<Layer * > parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, int enable_upscale, int dense_operation, string name, int dev, int mem)  : MLayer(name, dev, mem) {

    if(name.empty()) this->name = "HLSinf" + to_string(++total_layers);

    // HLSinf parameters
    this->H = h;   this->W = w;
    this->Ichannels = ichannels;  this->Ochannels = ochannels;
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
    this->enable_add        = enable_add;
    this->enable_upscale    = enable_upscale;
    this->dense_operation   = dense_operation;

    // HLSinf supports KHxKW = 1x1 by playing with paddings and zeroing filters
    if ((kh == 1) && (kw == 1)) {
      #ifdef DEBUG_FPGA
        printf("WARNING: Adjusting HLSinf layer to support 1x1 convolutions\n");
        if ((sh != 1) && (sw != 1)) printf("WARNING: 1x1 filter adjustment with strides different from 1\n");
      #endif
      this->PT = 0; this->PB = 2; this->PL = 0; this->PR = 2;
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
    if (enable_upscale) {HO = HO * 2; WO = WO * 2;}

    // Now, we create the tensors needed
    this->filter = new Tensor(vector<int>{ochannels, ichannels, KH, KW}, dev);
    this->bias = new Tensor(vector<int>{ochannels}, dev);
    this->batch_norm_values = new Tensor(vector<int>{ochannels*4}, dev);
    if (dense_operation) output = new Tensor({input->shape[0], ochannels * HO * WO}, dev); else output = new Tensor(vector<int>{input->shape[0], Ochannels, HO, WO}, dev);
    params.push_back(this->filter);
    params.push_back(this->bias);
    params.push_back(this->batch_norm_values);
}

int LHLSinf::get_trainable_params_count() {return 0;}

void LHLSinf::resize(int batch){output->resize(batch);}

void LHLSinf::forward() {
#ifdef cFPGA
    // The first time we perform a forward operation we need to check whether buffers have been created on the FPGA device
    // If not, then we allocate them and copy the tensor into the buffer, performing data type conversion if needed
    //
    // Filters
    if (filter->fpga_ptr == NULL) {
      if (hlsinf_filter_format == HLSINF_FP32) {
        // We simply create the buffer and copy the tensor into the buffer (no data type conversion needed)
        filter->fpga_ptr = fpga_create_memory(filter->size*sizeof(float));  
        fpga_copy_memory_to_fpga(filter->ptr, (cl::Buffer *)filter->fpga_ptr, filter->size*sizeof(float));
      } else if (hlsinf_filter_format == HLSINF_API8) {
        // Data conversion needed (FP32->API8)
        filter->fpga_ptr = fpga_create_memory(filter->size);  
        fpga_copy_memory_to_fpga_and_format(filter->ptr, (cl::Buffer *)filter->fpga_ptr, filter->size, HLSINF_FP32, HLSINF_API8);
      } else {
        printf("Error (HLSinf forward), filter format not supported\n");
        exit(1);
      }
    }
    // Bias
    if (bias->fpga_ptr == NULL) {
      if (hlsinf_bias_format == HLSINF_FP32) {
        // No need for data conversion (FP32->FP32), we allocate the buffer and copy the bias tensor there
        bias->fpga_ptr = fpga_create_memory(bias->size*sizeof(float));  
        fpga_copy_memory_to_fpga(bias->ptr, (cl::Buffer *)bias->fpga_ptr, bias->size*sizeof(float));
      } else if (hlsinf_bias_format == HLSINF_API32) {
        // Data conversion needed to API32 (FP32->API32)
        bias->fpga_ptr = fpga_create_memory(bias->size*4);  
        fpga_copy_memory_to_fpga_and_format(bias->ptr, (cl::Buffer *)bias->fpga_ptr, bias->size, HLSINF_FP32, HLSINF_API32);
      } else {
        printf("Error (HLSinf forward), bias format not supported\n");
        exit(1);
      }
    }
    // BatchNorm
    if (enable_batch_norm && (batch_norm_values->fpga_ptr == NULL)) {
      // BatchNorm values assumed to be always in FP32 (might not be the case!)
      batch_norm_values->fpga_ptr = fpga_create_memory(batch_norm_values->size*sizeof(float));  
      fpga_copy_memory_to_fpga(batch_norm_values->ptr, (cl::Buffer *)batch_norm_values->fpga_ptr, batch_norm_values->size*sizeof(float));
    }
    // Output buffer, the buffer size depends on the data type
    if (output->fpga_ptr == NULL) {
      if (hlsinf_output_format == HLSINF_FP32) {
        output->fpga_ptr = fpga_create_memory(output->size*sizeof(float));  
      } else if (hlsinf_output_format == HLSINF_API8) {
        output->fpga_ptr = fpga_create_memory(output->size);  
      } else if (hlsinf_output_format == HLSINF_APUI8) {
        output->fpga_ptr = fpga_create_memory(output->size);  
      } else {
        printf("Error (HLSinf forward), output format not supported\n");
        exit(1);
      }
    }
    // Now, we call the accelerator
    fpga_hlsinf(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_batch_norm, enable_maxp, enable_avgp,
		            enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_add, enable_stm, enable_upscale, 
                this->filter, this->bias, this->batch_norm_values, this->output);
#else
    msg("LHLSinf layer only available for FPGA", "LHLSinf::forward()");
#endif
}

void LHLSinf::backward() {msg("NotImplementedError", "LHLSinf::backward");}


Layer *LHLSinf::share(int c, int bs, vector<Layer *> p) {
 auto *n = new LHLSinf(p, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor,
                      enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_stm, enable_maxp, enable_avgp,
                      enable_batch_norm, enable_add, enable_upscale, dense_operation, "HLSinf_"+to_string(c)+this->name, this->dev, this->mem_level);

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
                    enable_batch_norm, enable_add, enable_upscale, dense_operation, name, todev, this->mem_level);
  n->orig = this;
  return n;
}

string LHLSinf::plot(int c) {
    string s;
    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    return s;
}
