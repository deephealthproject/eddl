
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/descriptors/tensor_descriptors.h"
#include "eddl/utils.h"

ExpandDescriptor::ExpandDescriptor(int size, int dev) : SelDescriptor(dev) {
    this->size = size;
}


void ExpandDescriptor::build(vector<int> ishape){
    // Get input/output shapes
    this->ishape = ishape;
    this->oshape = expand_shape(ishape, this->size);

    // Build indices
    this->build_indices();
}

void ExpandDescriptor::resize(int b){
//    // Update shapes
//    this->ishape[0] = b;
//    this->oshape[0] = b;

    // Build indices
    this->build_indices();
}

void ExpandDescriptor::build_indices(){
    // Delete previous allocations
    this->free_memory();

    // Compute index translation (output=>input)
    this->cpu_addresses = expand_indices(this->ishape, this->size);
}
