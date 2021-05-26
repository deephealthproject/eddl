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

#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/reductions/layer_reductions.h"


using namespace std;

int LRMax::total_layers = 0;

LRMax::LRMax(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem) : ReductionLayer(name, dev, mem) {
    // TODO: Implement
    if(name.empty()) this->name = "reduction_max" + to_string(++total_layers);

    input=l->output;
    this->axis=axis;
    this->keepdims=keepdims;


    // move all the axis +1 because 0 is for batch
    for(int i=0;i<axis.size();i++)
      axis[i]++;

    RD=new ReduceDescriptor(input,axis,"max",keepdims);

    output=RD->O;

    l->addchild(this);
    addparent(l);

}

void LRMax::forward(){
    reduction(RD);
}

void LRMax::backward(){
    reduction_back(RD);
}

// virtual
void LRMax::resize(int batch){
    RD->resize(batch);

}


Layer *LRMax::share(int c, int bs, vector<Layer *> p) {
    LRMax *n;
    n = new LRMax(p[0], axis, keepdims,  name, this->dev, this->mem_level);
    n->orig = this;
    return n;
}

Layer *LRMax::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRMax *n;
    n = new LRMax(p[0],axis, keepdims, "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
