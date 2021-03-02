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

#include "eddl/layers/reductions/layer_reductions.h"


using namespace std;

ReductionLayer::ReductionLayer(string name, int dev, int mem) : Layer(name, dev, mem) {
    binary=0;
}

ReductionLayer::~ReductionLayer(){
    delete RD;
}


void ReductionLayer::mem_delta(){
    if(this->delta == nullptr) {
        // Reserve parent's delta
        parent[0]->mem_delta();
        RD->ID = parent[0]->delta;

        delta = Tensor::zeros(RD->O->shape, RD->O->device);
        RD->D = delta;

        if(this->verbosity_level >= 2) {
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}


void ReductionLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void ReductionLayer::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}

string ReductionLayer::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
