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
#include "eddl/layers/generators/layer_generators.h"


using namespace std;

int LGauss::total_layers = 0;

LGauss::LGauss(float mean, float stdev, vector<int> size, string name, int dev, int mem): GeneratorLayer(name, dev, mem) {
    // TODO: Implement
    if(name.empty()) this->name = "generator_gauss" + to_string(++total_layers);

    this->mean=mean;
    this->stdev=stdev;
    this->size=size;

    size.insert(size.begin(),1);

    input=output=new Tensor(size, dev);
//    delta=new Tensor(size, dev);
}

void LGauss::forward(){
    output->rand_normal(mean, stdev);
}

void LGauss::backward(){

}


Layer *LGauss::share(int c, int bs, vector<Layer *> p) {
    // TODO: Implement
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LGauss::clone(int c, int bs, vector<Layer *> p, int todev) {
    // TODO: Implement
    LGauss *n;
    n = new LGauss(mean, stdev, size, "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
