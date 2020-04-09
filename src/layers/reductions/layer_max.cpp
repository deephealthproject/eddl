/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "layers/operators/layer_operators.h"
#include "layers/reductions/layer_reductions.h"


using namespace std;

int LRMax::total_layers = 0;

LRMax::LRMax(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem) : ReductionLayer(name, dev, mem) {
    // TODO: Implement
    if(name.empty()) this->name = "reduction_max" + to_string(++total_layers);

    input=l->output;

    RD=new ReduceDescriptor(input,axis,"max",keepdims);

    output=RD->O;
//    delta=RD->D;
//    RD->ID = l->delta;

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
    n = new LRMax(p[0], RD->axis, RD->keepdims, "share_" + to_string(c) + name, this->dev, this->mem_level);
    n->orig = this;
    return n;
}

Layer *LRMax::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRMax *n;
    n = new LRMax(p[0],RD->axis, RD->keepdims, "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
