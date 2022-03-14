/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "eddl/layers/da/layer_da.h"


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
    if (this->delta == nullptr) {
        // Reserve parent's delta AND assign it to this layer
        parent[0]->mem_delta();

        delta = parent[0]->delta;

        if(this->verbosity_level >= 2){
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}

void LDataAugmentation::free_delta(){
    // Not really needed, but I like to keep all the methods the same (ease the robustness of "copy-paste")
    if(this->delta != nullptr) {
        // Do not delete delta (points to parent)
        delta = nullptr;

        if(this->verbosity_level >= 2){
            std::cout << "Deleted delta for: " + this->name << std::endl;
        }
    }
}