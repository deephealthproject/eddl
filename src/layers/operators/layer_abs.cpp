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


using namespace std;

int LAbs::total_layers = 0;

/**
  @brief Computes the absolute value of a Layer

  @param l a Layer.
  @param name a name for the operation (predefined as 'abs+TotalAbsLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */
LAbs::LAbs(Layer *l, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    // Set default name
    if(name.empty()) this->name = "abs_" + to_string(++total_layers);

    input=l->output;

    mask=new Tensor(l->output->getShape(),dev);
    output=new Tensor(l->output->getShape(),dev);
//    delta=new Tensor(l->output->getShape(),dev);

    l->addchild(this);
    addparent(l);
}

void LAbs::forward(){
    Tensor::copy(parent[0]->output,output);
    output->abs_();

}

void LAbs::backward(){
    Tensor::sign(parent[0]->output,mask);
    Tensor::el_mult(delta,mask,parent[0]->delta,1);
}

void LAbs::resize(int b)
{
  Layer::resize(b);
  mask->resize(b);
}

Layer *LAbs::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LAbs::clone(int c, int bs, vector<Layer *> p, int todev) {
    LAbs *n = new LAbs(p[0],  name, todev, this->mem_level);
    n->orig = this;

    return n;
}
