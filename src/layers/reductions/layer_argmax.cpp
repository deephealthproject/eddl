/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/reductions/layer_reductions.h"


using namespace std;

int LRArgmax::total_layers = 0;

LRArgmax::LRArgmax(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem) : ReductionLayer(name, dev, mem) {
    if(name.empty()) this->name = "reduction_argmax" + to_string(++total_layers);

    input=l->output;
    this->keepdims=keepdims;

    // Move all the axis +1 because 0 is for batch
    this->axis = axis;  // Stored without batch
    for(int i=0;i<axis.size();i++) { axis[i]++; }

    // Build descriptor
    RD2 = new ReduceDescriptor2(axis, keepdims, dev);
    RD2->build(input->shape);

    // Create output tensor
    output = Tensor::empty(RD2->oshape, dev);

    l->addchild(this);
    addparent(l);

}

void LRArgmax::forward(){
    Tensor::argmax(input, output, RD2);
}

void LRArgmax::backward(){
    Tensor::argmax_d(this->delta, this->output, this->parent[0]->delta);
}

// virtual
void LRArgmax::resize(int batch){
    Layer::resize(batch);
    RD2->resize(batch);
}


Layer *LRArgmax::share(int c, int bs, vector<Layer *> p) {
    LRArgmax *n;
    n = new LRArgmax(p[0], axis, keepdims,  name, this->dev, this->mem_level);
    n->orig = this;
    return n;
}

Layer *LRArgmax::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRArgmax *n;
    n = new LRArgmax(p[0],axis, keepdims, "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
