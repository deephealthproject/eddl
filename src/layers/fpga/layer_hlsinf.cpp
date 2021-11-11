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
              int enable_relu, float relu_factor, int enable_clipping, int enable_shift, int pos_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, string name, int dev, int mem) :
              LHLSinf(vector<Layer*> {parent}, h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr,
              enable_relu, relu_factor, enable_clipping, enable_shift, pos_shift, enable_stm, enable_maxp, enable_avgp, enable_batch_norm,
              enable_add, {},  name, dev, mem) {  
};

LHLSinf::LHLSinf(Layer * parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int enable_shift, int pos_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, Tensor* batch_norm_val, string name, int dev, int mem) :
              LHLSinf(vector<Layer*> {parent}, h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr,
              enable_relu, relu_factor, enable_clipping, enable_shift, pos_shift, enable_stm, enable_maxp, enable_avgp, enable_batch_norm,
              enable_add, batch_norm_val, name, dev, mem) {  
};

LHLSinf::LHLSinf(vector<Layer * >parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int enable_shift, int pos_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, string name, int dev, int mem):
              LHLSinf(parent, h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr,
              enable_relu, relu_factor, enable_clipping, enable_shift, pos_shift, enable_stm, enable_maxp, enable_avgp, enable_batch_norm,
              enable_add, {}, name, dev, mem) {
};                

LHLSinf::LHLSinf(vector<Layer * > parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int enable_shift, int pos_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, Tensor* batch_norm_val, string name, int dev, int mem)  : MLayer(name, dev, mem) {

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
    this->enable_shift = enable_shift;
    this->pos_shift = pos_shift;
    this->enable_stm = enable_stm;
    this->enable_maxp = enable_maxp;
    this->enable_avgp = enable_avgp;
    this->enable_batch_norm = enable_batch_norm;
    this->enable_add = enable_add;

  printf("HLSinf: RELU %d RELU_FACTOR %f MAXP %d AVGP %d CLIPPING %d SHIFT %d BN %d ADD %d STM %d\n",
        enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, enable_shift, enable_batch_norm, enable_add, enable_stm);

    // we asllow K=1x1 by playing with paddings
    if ((kh == 1) && (kw == 1)) {
      printf("WARNING: Adjusting HLSinf layer to support 1x1 convolutions\n");
      if ((sh != 1) && (sw != 1)) printf("WARNING: 1x1 filter adjustment with strides different from 1\n");
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

    this->filter = new Tensor(vector<int>{ochannels, ichannels, KH, KW}, dev);
    this->bias = new Tensor(vector<int>{ochannels}, dev);

    this->input = parent[0]->output;

    params.push_back(this->filter);
    params.push_back(this->bias);

    Tensor *gK = new Tensor(vector<int>{ochannels, ichannels, KH, KW}, dev);
    Tensor *gbias = new Tensor(vector<int>{ochannels}, dev);

    gradients.push_back(gK);
    gradients.push_back(gbias);

    if(enable_add) this->input_add = parent[1]->output;

    if(enable_batch_norm) {
      this->batch_norm_values = new Tensor(vector<int>{ochannels*4}, DEV_CPU);
      tensor_padded(batch_norm_val,this->batch_norm_values);
    }

    output = new Tensor(vector<int>{input->shape[0], Ochannels, HO, WO}, dev);
    for (int i = 0; i < parent.size(); ++i) {
      parent[i]->addchild(this);
      addparent(parent[i]);
    }
}

// virtual
void LHLSinf::resize(int batch){
    output->resize(batch);
}


void LHLSinf::forward() {
#ifdef cFPGA
    fpga_hlsinf(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_batch_norm, enable_maxp, enable_avgp,
		enable_clipping, enable_shift, pos_shift, enable_add, enable_stm, this->filter, this->bias, this->batch_norm_values, this->output);
#else
    msg("LHLSinf layer only available for FPGA", "LHLSinf::forward()");
#endif
}

void LHLSinf::backward() {
    msg("NotImplementedError", "LHLSinf::backward");
}


Layer *LHLSinf::share(int c, int bs, vector<Layer *> p) {
 auto *n = new LHLSinf(p, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor,
                      enable_clipping, enable_shift, pos_shift, enable_stm, enable_maxp, enable_avgp,
                      enable_batch_norm, enable_add, batch_norm_values, "HLSinf_"+to_string(c)+this->name, this->dev, this->mem_level);

 return n;

}

Layer *LHLSinf::clone(int c, int bs, vector<Layer *> p, int todev) {
  auto *n = new LHLSinf(p, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor,
                    enable_clipping, enable_shift, pos_shift, enable_stm, enable_maxp, enable_avgp,
                    enable_batch_norm, enable_add, batch_norm_values, name, todev, this->mem_level);
  n->orig = this;

  return n;
}


string LHLSinf::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
