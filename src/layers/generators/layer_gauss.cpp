
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "../operators/layer_operators.h"
#include "layer_generators.h"


using namespace std;

int LGauss::total_layers = 0;

LGauss::LGauss(float mean, float stdev, initializer_list<int> &size, string name, int dev):LGauss(mean,stdev,vector<int>(size.begin(), size.end()),name,dev){}

LGauss::LGauss(float mean, float stdev, vector<int> size, string name, int dev): GeneratorLayer(name, dev) {
    // TODO: Implement
    if(name.empty()) this->name = "generator_gauss" + to_string(++total_layers);

    this->mean=mean;
    this->stdev=stdev;

    ////////////

}

void LGauss::forward(){
    // TODO: Implement
}

void LGauss::backward(){
  // TODO: Implement
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
