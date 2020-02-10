/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <utility>

#include "layer_da.h"


using namespace std;

LDataAugmentation::LDataAugmentation(Layer *parent, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    input = parent->output;
    // output = new Tensor(input->shape, dev);  // Build this on child
    delta = parent->delta;
}

LDataAugmentation::~LDataAugmentation(){
    delta = nullptr;
};


void LDataAugmentation::mem_delta(){
    // Reserve parent's delta AND assign it to this layer
    if (parent[0]->mem_level) {
        parent[0]->mem_delta();

        delta=parent[0]->delta;
    }
}

void LDataAugmentation::free_delta(){
    // Not really needed, but I like to keep all the methods the same (ease the robustness of "copy-paste")
    if(this->delta != nullptr) {
        delta = nullptr;
    }
}

//void LDataAugmentation::resize(int batch) {
//    output->resize(batch);
//    if (target!=nullptr) target->resize(batch);
//
//    // Ignore delta, as delta depends on another layer
//}