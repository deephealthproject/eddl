/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "../operators/layer_operators.h"
#include "layer_reductions.h"


using namespace std;

int LRMean::total_layers = 0;

/**
  @brief Computes the mean of elements across dimensions of a Layer

  @param l a Layer
  @param axis the dimensions to reduce. If NULL (the default), reduces all dimensions
  @param keepdims if true, retains reduced dimensions with length 1. Default False
  @param name a name for the operation (predefined as 'mean+TotaLRMeanLayers')
  @param dev which computing service utilize

  @returns the result of the logarithm operation over l

  Example:
  \verbatim
      # x contains [[1., 1.], [2., 2.]]
      eddl.Mean(x)  # 1.5
      eddl.Mean(x, 0)  # [1.5, 1.5]
      eddl.Mean(x, 1)  # [1.,  2.]
   \endverbatim

  */

LRMean::LRMean(Layer *l, vector <int> axis, bool keepdims, string name, int dev): ReductionLayer(name, dev) {
    if(name.empty()) this->name = "reduction_mean" + to_string(++total_layers);

    input=l->output;

    RD=new ReduceDescriptor(input,axis,"mean",keepdims);

    output=RD->O;
    delta=RD->D;
    RD->ID = l->delta;

    l->addchild(this);
    addparent(l);

}

void LRMean::forward(){
    reduction(RD);
}

void LRMean::backward(){
    reduction_back(RD);
}

// virtual
void LRMean::resize(int batch){
    RD->resize(batch);
}


Layer *LRMean::share(int c, int bs, vector<Layer *> p) {
  LRMean *n;
  n = new LRMean(p[0], RD->axis, RD->keepdims, "share_" + to_string(c) + name,dev);
  n->orig = this;
  return n;
}

Layer *LRMean::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRMean *n;
    n = new LRMean(p[0],RD->axis, RD->keepdims, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
