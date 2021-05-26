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

#include "eddl/layers/core/layer_core.h"


using namespace std;

int LUpSampling3D::total_layers = 0;

LUpSampling3D::LUpSampling3D(Layer *parent, vector<int> new_shape, bool reshape, WrappingMode da_mode, float cval, TransformationMode coordinate_transformation_mode, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "upsampling3d" + to_string(++total_layers);

    input = parent->output;
    if (reshape){
        output = new Tensor({this->input->shape[0], this->input->shape[1], new_shape[0], new_shape[1], new_shape[2]}, dev);
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


void LUpSampling3D::forward() {
    Tensor::scale3d(this->input, this->output, this->new_shape, this->da_mode, this->cval, this->coordinate_transformation_mode);
}

void LUpSampling3D::backward() {
    Tensor::scale3d_back(parent[0]->delta, this->delta, this->new_shape, this->da_mode, this->cval, this->coordinate_transformation_mode);
}


Layer *LUpSampling3D::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LUpSampling3D(p[0], this->new_shape, this->reshape, this->da_mode, this->cval, this->coordinate_transformation_mode, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LUpSampling3D::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LUpSampling3D(p[0], this->new_shape, this->reshape, this->da_mode, this->cval,  this->coordinate_transformation_mode, this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LUpSampling3D::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
