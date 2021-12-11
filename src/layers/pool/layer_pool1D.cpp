/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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

    this->input = parent->output;

    // Reshape the 2D input to a 3D tensor
    vector<int> in_shape = this->input->getShape();
    in_shape.push_back(1);
    this->input_reshaped = new Tensor(in_shape, this->input);

    this->pd = D;
    this->pd->build(input_reshaped);

    // Reshape the 3D output from conv to a 2D tensor
    vector<int> out_shape = this->pd->O->getShape();
    out_shape.pop_back();
    this->output = new Tensor(out_shape, this->pd->O);

    parent->addchild(this);
    addparent(parent);
}

LPool1D::~LPool1D(){
    delete this->pd->O; // TO BE REVIEWED
    delete this->pd;
    delete this->input_reshaped;
}

void LPool1D::mem_delta(){
    if(this->delta == nullptr) {
        // Reserve parent's delta
        parent[0]->mem_delta();
        pd->ID = parent[0]->delta;

        delta = Tensor::zeros(output->shape, output->device);
        pd->D = delta; // new Tensor(pd->O->shape, delta); TO BE REVIEWED

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
