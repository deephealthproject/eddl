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
#include <utility>

#include "eddl/layers/core/layer_core.h"
#include "eddl/hardware/fpga/fpga_hw.h"

using namespace std;

int LTransform::total_layers = 0;

LTransform::LTransform(Layer *parent, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    // Set default name
    if(name.empty()) this->name = "transform_" + to_string(++total_layers);

    int CPI = 0;
#ifdef cFPGA
    CPI = hlsinf_cpi;
#endif

    if(!CPI) msg("Error: LTransform layer with CPI parameter equal to 0 ");

    // Set input
    input=parent->output;

    // Set mode
    this->copy_cpu_to_fpga = copy_cpu_to_fpga;
    this->copy_fpga_to_cpu = copy_fpga_to_cpu;
    this->transform = transform;

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

    // output tensor
    output=new Tensor(oshape, dev);

    // parents-childs relationship
    parent->addchild(this);
    addparent(parent);
}

LTransform::~LTransform() {}

void LTransform::resize(int b){
    Layer::resize(b);
}

void LTransform::forward(){
    if (layer_disabled) return;
    tensorNN::transform(this->input, this->output, this->copy_cpu_to_fpga, this->copy_fpga_to_cpu, this->transform);
}

void LTransform::backward(){
    printf("Error, Transform layer can be used only for inference\n");
    exit(1);
}

Layer *LTransform::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LTransform::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LTransform(p[0], copy_cpu_to_fpga, copy_fpga_to_cpu, transform, name, todev, this->mem_level);
    n->orig = this;
    return n;
}
