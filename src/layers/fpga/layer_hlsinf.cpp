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


using namespace std;

int LHLSinf::total_layers = 0;

LHLSinf::LHLSinf(Layer *parent, string name, int dev, int mem, int KH, int KW, int SH, int SW, int PH, int PW, int enable_relu, int enable_maxpooling, int enable_add, int enable_stm) : LinLayer(name, dev, mem) {

    if(name.empty()) this->name = "HLSinf" + to_string(++total_layers);

    this->KH = KH;
    this->KW = KW;
    this->SH = SH;
    this->SW = SW;
    this->PH = PH;
    this->PW = PW;
    this->enable_relu = enable_relu;
    this->enable_maxpooling = enable_maxpooling;
    this->enable_add = enable_add;
    this->enable_stm = enable_stm;

    this->input = parent->output;
    output = new Tensor(input->shape, dev);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LHLSinf::resize(int batch){
    output->resize(batch);
}


void LHLSinf::forward() {
    //fpga_hlsinf(this->input, KH, KW, SH, SW, PH, PW, enable_relu, enable_maxpooling, enable_add, enable_stm, this->filters, this->bias);
    printf("HLSinf forward not implemented yet\n"); exit(1);
}

void LHLSinf::backward() {
    msg("NotImplementedError", "LHLSinf::backward");
}


Layer *LHLSinf::share(int c, int bs, vector<Layer *> p) {
    msg("NotImplementedError", "LHLSinf::share");
}

Layer *LHLSinf::clone(int c, int bs, vector<Layer *> p, int todev) {
  msg("NotImplementedError", "LHLSinf::share");
       	// auto *n = new LHLSinf(this->parent, name, todev, this->mem_level, KH, KW, SH, SW, PH, PW, enable_relu, enable_maxpooling, enable_add, enable_stm);

//    return n;
}


string LHLSinf::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
