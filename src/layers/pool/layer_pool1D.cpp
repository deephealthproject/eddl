/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/pool/layer_pool.h"


using namespace std;


int LPool1D::total_layers = 0;

LPool1D::LPool1D(Layer *parent, PoolDescriptor *D, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 3) msg("LPool1D only works over 3D tensors", "LPool1D::LPool1D");
    if(name.empty()) this->name = "pool1D" + to_string(++total_layers);

    input = parent->output;

    // Reshape the 2D input to a 3D tensor
    vector<int> in_shape = input->getShape();
    in_shape.push_back(1);
    input_reshaped = new Tensor(in_shape, input);

    pd = D;
    pd->build(input_reshaped);

    // Reshape the 3D output from conv to a 2D tensor
    vector<int> out_shape = pd->O->getShape();
    out_shape.pop_back();
    output = new Tensor(out_shape, pd->O);

//  delta = pd->D;
//  pd->ID = parent->delta;

    parent->addchild(this);
    addparent(parent);
}

LPool1D::~LPool1D(){
//    delete pd;
}

void LPool1D::mem_delta(){
    if(this->delta == nullptr) {
        // Reserve parent's delta
        parent[0]->mem_delta();
        pd->ID = parent[0]->delta;

        delta = Tensor::zeros(output->shape, output->device);
        pd->D = new Tensor(pd->O->shape, delta);

        if(this->verbosity_level >= 2) {
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}


void LPool1D::resize(int batch){
    // Resize but keeping the pointer to the input before the reshape
    input_reshaped->resize(batch, input->ptr);

    pd->resize(batch);

    // Resize but keeping the pointer to the output of the descriptor
    output->resize(batch, pd->O->ptr);
}
