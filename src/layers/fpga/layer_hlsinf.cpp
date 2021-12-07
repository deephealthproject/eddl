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

void print_tensor11(Tensor *T) {
  int d1_max = 2;
  int d2_max = 4;
  int d3_max = 4;
  printf("shape %d\n", T->ndim);
  if (T->ndim==4) {
    for (int d0=0; d0<T->shape[0]; d0++) {
    for (int d1=0; d1<T->shape[1]; d1++) {
    for (int d2=0; d2<T->shape[2]; d2++) {
    for (int d3=0; d3<T->shape[3]; d3++) {
      int a = (d0 * T->shape[1] * T->shape[2] * T->shape[3]) + (d1 * T->shape[2] * T->shape[3]) + (d2 * T->shape[3]) + d3;
      printf("%f ", T->ptr[a]);
    }
    printf("\n\n");
    }
    printf("\n---\n");
    }
    printf("\n--\n--\n");
    }
  }  else if(T->ndim==2) {
    for (int d0=0; d0<T->shape[0]; d0++) {
    for (int d1=0; d1<T->shape[1]; d1++) {
      int a = (d0 * T->shape[1]) + d1;
      printf("%f ", T->ptr[a]);
    }
    printf("\n\n");
    }

  } else if(T->ndim==1) {
    for (int d0=0; d0<T->shape[0]; d0++) {
      printf("%f ", T->ptr[d0]);
    }
    printf("\n\n");
    }
}

LHLSinf::LHLSinf(Layer * parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, int enable_upscale, int dense_operation, string name, int dev, int mem) :
              LHLSinf(vector<Layer*> {parent}, h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr,
              enable_relu, relu_factor, enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_stm, enable_maxp, enable_avgp, enable_batch_norm,
              enable_add, enable_upscale, dense_operation, name, dev, mem) {  
};

LHLSinf::LHLSinf(vector<Layer * > parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, int enable_upscale, int dense_operation, string name, int dev, int mem)  : MLayer(name, dev, mem) {

    if(name.empty()) this->name = "HLSinf" + to_string(++total_layers);

    this->H = h;
    this->W = w;
    this->Ichannels = ichannels;
    this->Ochannels = ochannels;
    this->KH = 3; //kh;
    this->KW = 3; //kw;
    this->SH = sh;
    this->SW = sw;
    this->PT = pt;
    this->PB = pb;
    this->PL = pl;
    this->PR = pr;
    this->enable_relu = enable_relu;
    this->relu_factor = relu_factor;
    this->enable_clipping = enable_clipping;
    this->min_clip = min_clip;
    this->max_clip = max_clip;
    this->enable_shift = enable_shift;
    this->pos_shift = pos_shift;
    this->dir_shift = dir_shift;
    this->enable_stm = enable_stm;
    this->enable_maxp = enable_maxp;
    this->enable_avgp = enable_avgp;
    this->enable_batch_norm = enable_batch_norm;
    this->enable_add = enable_add;
    this->enable_upscale = enable_upscale;
    this->dense_operation = dense_operation;

  //printf("HLSinf: I %d O %d KH %d KW %d SH %d SH %d PT %d PB %d PL %d PR %d RELU %d RELU_FACTOR %f MAXP %d AVGP %d CLIPPING %d SHIFT %d BN %d ADD %d STM %d dense_operation %d\n",
  //      Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, enable_shift, enable_batch_norm, enable_add, enable_stm, dense_operation);

    // we allow K=1x1 by playing with paddings
    if ((kh == 1) && (kw == 1)) {
      //printf("WARNING: Adjusting HLSinf layer to support 1x1 convolutions\n");
      //if ((sh != 1) && (sw != 1)) printf("WARNING: 1x1 filter adjustment with strides different from 1\n");
      this->PT = 0;
      this->PB = 2;
      this->PL = 0;
      this->PR = 2;
    }
    
    int HO = (H + PT + PB - KH + SH) / SH;
    int WO = (W + PL + PR - KW + SW) / SW;
    if (enable_maxp || enable_avgp) {
      HO = HO / 2;
      WO = WO / 2;
    }
    if (enable_upscale) {HO = HO * 2; WO = WO * 2;}

    this->filter = new Tensor(vector<int>{ochannels, ichannels, KH, KW}, dev);
    this->bias = new Tensor(vector<int>{ochannels}, dev);
    this->input = parent[0]->output;

    params.push_back(this->filter);
    params.push_back(this->bias);

    if(enable_add) this->input_add = parent[1]->output;

    this->batch_norm_values = new Tensor(vector<int>{ochannels*4}, dev);
    params.push_back(this->batch_norm_values);

    if (dense_operation) {
      output = new Tensor({input->shape[0], ochannels * HO * WO}, dev);
    } else {
      output = new Tensor(vector<int>{input->shape[0], Ochannels, HO, WO}, dev);
    }
    for (int i = 0; i < parent.size(); ++i) {
      parent[i]->addchild(this);
      addparent(parent[i]);
    }
}

int LHLSinf::get_trainable_params_count() {return 0;}

// virtual
void LHLSinf::resize(int batch){
    output->resize(batch);
}

void LHLSinf::forward() {
#ifdef cFPGA
    if (filter->fpga_ptr == NULL) {
      if (hlsinf_filter_format == HLSINF_FP32) {
        filter->fpga_ptr = fpga_create_memory(filter->size*sizeof(float));  
        fpga_copy_memory_to_fpga(filter->ptr, (cl::Buffer *)filter->fpga_ptr, filter->size*sizeof(float));
      } else if (hlsinf_filter_format == HLSINF_API8) {
        filter->fpga_ptr = fpga_create_memory(filter->size);  
        fpga_copy_memory_to_fpga_and_format(filter->ptr, (cl::Buffer *)filter->fpga_ptr, filter->size, HLSINF_FP32, HLSINF_API8);
      } else {
        printf("Error (HLSinf forward), filter format not supported\n");
        exit(1);
      }
    }
    if (bias->fpga_ptr == NULL) {
      if (hlsinf_bias_format == HLSINF_FP32) {
        bias->fpga_ptr = fpga_create_memory(bias->size*sizeof(float));  
        fpga_copy_memory_to_fpga(bias->ptr, (cl::Buffer *)bias->fpga_ptr, bias->size*sizeof(float));
      } else if (hlsinf_bias_format == HLSINF_API32) {
        bias->fpga_ptr = fpga_create_memory(bias->size*4);  
        fpga_copy_memory_to_fpga_and_format(bias->ptr, (cl::Buffer *)bias->fpga_ptr, bias->size, HLSINF_FP32, HLSINF_API32);
      } else {
        printf("Error (HLSinf forward), bias format not supported\n");
        exit(1);
      }
    }
    if (enable_batch_norm && (batch_norm_values->fpga_ptr == NULL)) {
      batch_norm_values->fpga_ptr = fpga_create_memory(batch_norm_values->size*sizeof(float));  
      fpga_copy_memory_to_fpga(batch_norm_values->ptr, (cl::Buffer *)batch_norm_values->fpga_ptr, batch_norm_values->size*sizeof(float));
    }
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

    fpga_hlsinf(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_batch_norm, enable_maxp, enable_avgp,
		enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_add, enable_stm, enable_upscale, this->filter, this->bias, this->batch_norm_values, this->output);
#else
    msg("LHLSinf layer only available for FPGA", "LHLSinf::forward()");
#endif
}

void LHLSinf::backward() {
    msg("NotImplementedError", "LHLSinf::backward");
}


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
