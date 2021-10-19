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
#include <utility>

#include "eddl/layers/core/layer_core.h"
#include "eddl/hardware/fpga/fpga_hw.h"

using namespace std;

int LTransform::total_layers = 0;

/**
  @brief Transforms input data to an specific output data format

  @param l a Layer.
  @param name a name for the operation
  @param dev which computing service utilize

  @returns 

  */
LTransform::LTransform(Layer *parent, int mode, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    // Set default name
    if(name.empty()) this->name = "transform_" + to_string(++total_layers);

    int CPI = 0;

    #ifdef cFPGA
    CPI = k_conv2d_cpi;
    #endif

    if(!CPI) {
        msg("Error: LTransform layer with CPI parameter equal to 0 ");
    }

    // Set input
    input=parent->output;

    // Set mode
    this->mode = mode;

    // Set flow tensors
    int ndim = input->ndim;
    int B = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];

    // output channels must be multiple of CPI
    vector<int> oshape(input->shape);
    int Cout = ((C + CPI - 1) / CPI) * CPI;
    oshape[1] = Cout;

//    printf("out: ndim %d B %d C %d H %d W %d\n", ndim, B, Cout, H, W);

    output=new Tensor(oshape, dev);

    parent->addchild(this);
    addparent(parent);
}

LTransform::~LTransform(){
    delete sd;
}

void LTransform::resize(int b){
    Layer::resize(b);
    sd->resize(b); // The batch is ignored
}

void LTransform::forward(){
    tensorNN::transform(this->input, this->output, this->mode);
}

void LTransform::backward(){
    printf("Error, Transform layer can be used only for inference\n");
    exit(1);
}

Layer *LTransform::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LTransform::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LTransform(p[0], mode, name, todev, this->mem_level);
    n->orig = this;
    return n;
}
