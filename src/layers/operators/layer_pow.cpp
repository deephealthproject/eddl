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


using namespace std;

int LPow::total_layers = 0;


LPow::LPow(Layer *l, float k, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "pow_" + to_string(++total_layers);

    this->exponent = k;

    input = l->output;
    output = Tensor::empty_like(input);

    l->addchild(this);
    addparent(l);
}

void LPow::forward(){
    Tensor::pow(parent[0]->output, output, this->exponent);
}

void LPow::backward(){
    // f(x) = x^3 => f'(x) = 3*x^(3-1)
    Tensor::pow(this->input, parent[0]->delta, this->exponent-1.0f);  // tmp=x^(3-1)
    parent[0]->delta->mult_(this->exponent); // tmp=3*[x^(3-1)]
    Tensor::mult(delta, parent[0]->delta, parent[0]->delta);// tmp=delta*[3*x^(3-1)]
}

Layer *LPow::share(int c, int bs, vector<Layer *> p) {
    return clone(c, bs, p, this->dev);
}

Layer *LPow::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LPow(p[0], this->exponent, this->name, todev, this->mem_level);
    n->orig = this;
    return n;
}