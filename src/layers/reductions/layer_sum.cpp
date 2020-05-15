/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
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

int LRSum::total_layers = 0;

LRSum::LRSum(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem) : ReductionLayer(name, dev, mem) {
    // TODO: Implement
    if(name.empty()) this->name = "reduction_sum" + to_string(++total_layers);

    input=l->output;
    // move all the axis +1 because 0 is for batch
    for(int i=0;i<axis.size();i++)
      axis[i]++;
      
    RD=new ReduceDescriptor(input,axis,"sum",keepdims);

    output=RD->O;
//    delta=RD->D;
//    RD->ID = l->delta;

    l->addchild(this);
    addparent(l);

}

void LRSum::forward(){
    reduction(RD);
}

void LRSum::backward(){
    reduction_back(RD);
}

// virtual
void LRSum::resize(int batch){
    RD->resize(batch);

}


Layer *LRSum::share(int c, int bs, vector<Layer *> p) {
    LRSum *n;
    n = new LRSum(p[0], RD->axis, RD->keepdims,  name, this->dev, this->mem_level);
    n->orig = this;
    return n;
}

Layer *LRSum::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRSum *n;
    n = new LRSum(p[0],RD->axis, RD->keepdims, "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
