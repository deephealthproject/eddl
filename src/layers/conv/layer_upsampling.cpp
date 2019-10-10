/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_conv.h"


using namespace std;

int LUpSampling::total_layers = 0;

LUpSampling::LUpSampling(Layer *parent, const vector<int> &size, string interpolation, string name, int dev) : LinLayer(name, dev) {
    this->size = size;
    this->interpolation = interpolation;
    if(name.empty()) this->name = "upsampling" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor(vector<int>{input->shape[0], input->shape[1], input->shape[2]*size[0], input->shape[3]*size[1]}, dev);
    delta = new Tensor(output->getShape(), dev);

    parent->addchild(this);
    addparent(parent);
}



void LUpSampling::resize(int batch){
    Layer::resize(batch);
}

void LUpSampling::forward() {
    //Repeats the rows and columns of the data by size[0] and size[1] respectively.
    repeat_nn(this->input, this->output, this->size);
}

void LUpSampling::backward() {
    d_repeat_nn(delta, parent[0]->delta, this->size);
}

Layer *LUpSampling::share(int c, int bs, vector<Layer *> p) {
    LUpSampling *n = new LUpSampling(p[0], this->size, this->interpolation,
            "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LUpSampling::clone(int c, int bs, vector<Layer *> p, int todev) {
    LUpSampling *n = new LUpSampling(p[0], this->size, this->interpolation, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}

string LUpSampling::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=red,shape=box]";

    return s;
}
