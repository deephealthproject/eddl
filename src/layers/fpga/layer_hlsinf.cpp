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

using namespace std;

int LHLSinf::total_layers = 0;

LHLSinf::LHLSinf(Layer * parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_maxp, int enable_avgp, int enable_clipping, int enable_shift, int pos_shift,
              int enable_add, int enable_stm, string name, int dev, int mem) :
              LHLSinf(vector<Layer*> {parent}, h, w, ichannels, ochannels, kh, kw, sh, sw, pt, pb, pl, pr,
              enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, enable_shift, pos_shift,
              enable_add, enable_stm, name, dev, mem) {  
};


LHLSinf::LHLSinf(vector<Layer *> parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
                    int enable_relu, float relu_factor, int enable_maxp, int enable_avgp, int enable_clipping, int enable_shift, int pos_shift,
                    int enable_add, int enable_stm, string name, int dev, int mem) : MLayer(name, dev, mem) {

    if(name.empty()) this->name = "HLSinf" + to_string(++total_layers);

    this->H = h;
    this->W = w;
    this->Ichannels = ichannels;
    this->Ochannels = ochannels;
    this->KH = kh;
    this->KW = kw;
    this->SH = sh;
    this->SW = sw;
    this->PT = pt;
    this->PB = pb;
    this->PL = pl;
    this->PR = pr;
    this->enable_relu = enable_relu;
    this->relu_factor = relu_factor;
    this->enable_maxp = enable_maxp;
    this->enable_avgp = enable_avgp;
    this->enable_clipping = enable_clipping;
    this->enable_shift = enable_shift;
    this->pos_shift = pos_shift;
    this->enable_add = enable_add;
    this->enable_stm = enable_stm;
    
    int HO = (H + PT + PB - KH + SH) / SH;
    int WO = (W + PL + PR - KW + SW) / SW;
    this->filter = new Tensor(vector<int>{ochannels, kh, kw, ichannels}, dev);
    this->bias = new Tensor(vector<int>{ochannels}, dev);

  //      printf("HLSinf\n filter \n");
  //      for (int i = 0 ;i < filter->shape.size() ;i++)
  //  printf(" %d ", filter->shape[i]);
  //  printf("\n bias \n");
  // for (int i = 0; i < bias->shape.size() ;i++)
  //  printf(" %d ", bias->shape[i]);
  //  printf("\n Ochann %d , HO %d WO %d \n", Ochannels, HO, WO);

    this->input = parent[0]->output;
    
    //params.push_back(this->filter);
    //params.push_back(this->bias);

    if(enable_add) this->input_add = parent[1]->output;
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
       	fpga_hlsinf(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_maxp, enable_avgp,
		                enable_clipping, enable_shift, pos_shift, enable_add, enable_stm, this->filter, this->bias, this->output);
}

void LHLSinf::backward() {
    msg("NotImplementedError", "LHLSinf::backward");
}


Layer *LHLSinf::share(int c, int bs, vector<Layer *> p) {
   printf("\n\n\n HLSINF SHARE\n\n\n");

 auto *n = new LHLSinf(p, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, enable_shift, pos_shift,
		 enable_add, enable_stm, "HLSinf_"+to_string(c)+this->name, this->dev, this->mem_level);
 return n;

}

Layer *LHLSinf::clone(int c, int bs, vector<Layer *> p, int todev) {
     printf("\n\n\n HLSINF clone\n\n\n");

  auto *n = new LHLSinf(p, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, enable_shift, pos_shift,
		  enable_add, enable_stm, name, todev, this->mem_level);
  n->orig = this;

  return n;
}


string LHLSinf::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
