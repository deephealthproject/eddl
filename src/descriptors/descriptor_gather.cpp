
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/descriptors/tensor_descriptors.h"
#include "eddl/utils.h"

GatherDescriptor::GatherDescriptor(const vector<int>& dims, int dev) : SelDescriptor(dev) {
    this->dims = vector<int>(dims);
}


void GatherDescriptor::build(vector<int> ishape){
    // Get input/output shapes
    this->ishape = ishape;
    this->oshape = permute_shape(ishape, this->dims);

    // Build indices
    this->build_indices();
}

void GatherDescriptor::resize(int b){
//    // Update shapes
//    this->ishape[0] = b;
//    this->oshape[0] = b;

    // Build indices
    this->build_indices();
}

void GatherDescriptor::build_indices(){
    // Delete previous allocations
    this->free_memory();

    // Compute index translation (output=>input)
    this->cpu_addresses = permute_indices(this->ishape, this->dims);
}
