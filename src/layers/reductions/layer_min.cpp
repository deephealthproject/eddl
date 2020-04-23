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

#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/reductions/layer_reductions.h"


using namespace std;

int LRMin::total_layers = 0;

LRMin::LRMin(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem) : ReductionLayer(name, dev, mem) {
    // TODO: Implement
    if(name.empty()) this->name = "reduction_min" + to_string(++total_layers);

    input=l->output;

    RD=new ReduceDescriptor(input,axis,"min",keepdims);

    output=RD->O;
//    delta=RD->D;
//    RD->ID = l->delta;

    l->addchild(this);
    addparent(l);

}

void LRMin::forward(){
    reduction(RD);
}

void LRMin::backward(){
    reduction_back(RD);
}

// virtual
void LRMin::resize(int batch){
    RD->resize(batch);
    
}


Layer *LRMin::share(int c, int bs, vector<Layer *> p) {
    LRMin *n;
    n = new LRMin(p[0], RD->axis, RD->keepdims,  name, this->dev, this->mem_level);
    n->orig = this;
    return n;
}

Layer *LRMin::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRMin *n;
    n = new LRMin(p[0],RD->axis, RD->keepdims, "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
