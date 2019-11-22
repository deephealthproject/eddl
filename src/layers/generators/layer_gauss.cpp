/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "../operators/layer_operators.h"
#include "layer_generators.h"


using namespace std;

int LGauss::total_layers = 0;

LGauss::LGauss(float mean, float stdev, vector<int> size, string name, int dev): GeneratorLayer(name, dev) {
    // TODO: Implement
    if(name.empty()) this->name = "generator_gauss" + to_string(++total_layers);

    this->mean=mean;
    this->stdev=stdev;
    this->size=size;

    size.insert(size.begin(),1);

    input=output=new Tensor(size,dev);
    delta=new Tensor(size,dev);

    ////////////

}

void LGauss::forward(){
    output->rand_normal(mean, stdev);
}

void LGauss::backward(){

}

void LGauss::resize(int b){
    output->resize(b);
    delta->resize(b);
}


Layer *LGauss::share(int c, int bs, vector<Layer *> p) {
    // TODO: Implement
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LGauss::clone(int c, int bs, vector<Layer *> p, int todev) {
    // TODO: Implement
    LGauss *n;
    n = new LGauss(mean, stdev, size, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
