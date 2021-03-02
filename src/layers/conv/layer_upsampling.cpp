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

#include "eddl/layers/conv/layer_conv.h"


using namespace std;

int LUpSampling::total_layers = 0;

LUpSampling::LUpSampling(Layer *parent, const vector<int> &size, string interpolation, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    this->size = size;
    this->interpolation = interpolation;

    if(name.empty()) this->name = "upsampling" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor(vector<int>{input->shape[0], input->shape[1], input->shape[2]*size[0], input->shape[3]*size[1]}, dev);
//    delta = new Tensor(output->shape, dev);

    parent->addchild(this);
    addparent(parent);
}


void LUpSampling::forward() {
    //Repeats the rows and columns of the data by size[0] and size[1] respectively.
    tensorNN::repeat_nn(this->input, this->output, this->size);
}

void LUpSampling::backward() {
    tensorNN::d_repeat_nn(delta, parent[0]->delta, this->size);
}

Layer *LUpSampling::share(int c, int bs, vector<Layer *> p) {
    LUpSampling *n = new LUpSampling(p[0], this->size, this->interpolation, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LUpSampling::clone(int c, int bs, vector<Layer *> p, int todev) {
    LUpSampling *n = new LUpSampling(p[0], this->size, this->interpolation,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}

string LUpSampling::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
