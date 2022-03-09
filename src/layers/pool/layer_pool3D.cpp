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

#include "eddl/layers/pool/layer_pool.h"


using namespace std;


int LPool3D::total_layers = 0;

LPool3D::LPool3D(Layer *parent, PoolDescriptor3D *D, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 5) msg("LPool3D only works over 5D tensors", "LPool3D::LPool3D");
    if(name.empty()) this->name = "pool3d" + to_string(++total_layers);

    input = parent->output;
    pd = D;

    pd->build(input);

    output = pd->O;

    parent->addchild(this);
    addparent(parent);
}

LPool3D::~LPool3D(){
   delete pd;
}

void LPool3D::mem_delta(){
    if(this->delta == nullptr) {
        // Reserve parent's delta
        parent[0]->mem_delta();
        pd->ID = parent[0]->delta;

        delta = Tensor::zeros(pd->O->shape, pd->O->device);
        pd->D = delta;

        if(this->verbosity_level >= 2) {
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}


void LPool3D::resize(int batch){
    pd->resize(batch);
    
}
