/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/core/layer_core.h"


using namespace std;

int LResize::total_layers = 0;

LResize::LResize(Layer *parent, vector<int> new_shape, bool reshape, WrappingMode da_mode, float cval, TransformationMode coordinate_transformation_mode, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "resize" + to_string(++total_layers);

    input = parent->output;
    if (reshape){
        output = new Tensor({this->input->shape[0], this->input->shape[1], new_shape[0], new_shape[1]}, dev);
    }else{
        output = new Tensor(input->shape, dev);
    }

    // Params
    this->new_shape = new_shape;
    this->reshape = reshape;
    this->cval = cval;
    this->da_mode = da_mode;
    this->coordinate_transformation_mode = coordinate_transformation_mode;

    parent->addchild(this);
    addparent(parent);

}


void LResize::forward() {
    Tensor::scale(this->input, this->output, this->new_shape, this->da_mode, this->cval, this->coordinate_transformation_mode);
}

void LResize::backward() {
    Tensor::scale_back(parent[0]->delta, this->delta, this->new_shape, this->da_mode, this->cval, this->coordinate_transformation_mode);
}


Layer *LResize::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LResize(p[0], this->new_shape, this->reshape, this->da_mode, this->cval, this->coordinate_transformation_mode, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LResize::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LResize(p[0], this->new_shape, this->reshape, this->da_mode, this->cval,  this->coordinate_transformation_mode, this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LResize::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
