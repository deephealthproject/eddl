/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/core/layer_core.h"


using namespace std;

int LRepeat::total_layers = 0;

LRepeat::LRepeat(Layer *parent, const vector<unsigned int>& repeats, unsigned int axis, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "repeat" + to_string(++total_layers);

    // Set input
    input = parent->output;

    // Check axis values
    if(axis<0 || axis > input->ndim-1){
        msg("The axis must be a number between 0 and the maximum dimension of the tensor", "LRepeat::LRepeat");
    }

    // Check that there are enough values in
    if(repeats.size()!=input->shape[axis]){
        msg("The size of 'repeats' (" + std::to_string(repeats.size()) + ") must equal the size the the dimension to repeat " + std::to_string(input->shape[axis]) + ")", "LRepeat::LRepeat");
    }

    // Get input shape, but making sure the batch dimension is 1
    vector<int> input_shape_single_batch(input->shape.begin()+1, input->shape.end());
    input_shape_single_batch.insert(input_shape_single_batch.begin(), 1);

    // Build descriptor (batch must equal to "1")
    this->rd = new RepeatDescriptor(repeats, axis, dev);
    this->rd->build(input_shape_single_batch);

    // Set output tensors
    output=new Tensor(this->rd->oshape, dev);

    // Set parent
    parent->addchild(this);
    addparent(parent);
}

// This constructor is also in the API with sanity checks
LRepeat::LRepeat(Layer *parent, unsigned int repeats, unsigned int axis, string name, int dev, int mem) : LRepeat(parent, vector<unsigned int>(parent->output->shape[axis], repeats), axis, name, dev, mem) {}

LRepeat::~LRepeat(){
    delete rd;
}

void LRepeat::forward(){
    tensorNN::select(this->input, this->output, this->rd);
//    Tensor::repeat(this->input, this->rd->vrepeats, this->rd->axis, this->output);
}

void LRepeat::backward(){
    tensorNN::select_back(this->delta, this->parent[0]->delta, this->rd);
//    Tensor::repeat(this->input, this->rd->vrepeats, this->rd->axis, this->output, true);
}

Layer *LRepeat::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LRepeat(p[0], this->rd->vrepeats, this->rd->axis, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LRepeat::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LRepeat(p[0], this->rd->vrepeats, this->rd->axis, this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LRepeat::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
