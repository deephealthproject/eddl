/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/da/layer_da.h"


using namespace std;

int LCutout::total_layers = 0;

LCutout::LCutout(Layer *parent, vector<int> from_coords, vector<int> to_coords, float cval, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "cutout" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    // Params
    this->from_coords = from_coords;
    this->to_coords = to_coords;
    this->cval = cval;

    parent->addchild(this);
    addparent(parent);

}


void LCutout::forward() {
    Tensor::cutout(this->input, this->output, this->from_coords, this->to_coords, this->cval);
}

void LCutout::backward() {

}


Layer *LCutout::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LCutout(p[0], this->from_coords, this->to_coords, this->cval, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LCutout::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LCutout(p[0], this->from_coords, this->to_coords, this->cval,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LCutout::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
