
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "tensor_descriptors.h"
#include "../utils.h"

PermuteDescriptor::PermuteDescriptor(const vector<int>& dims) : SelDescriptor() {
    addresses=nullptr;
    this->dims = vector<int>(dims);
}


void PermuteDescriptor::build(vector<int> ishape){
    // Get input/output shapes
    this->ishape = ishape;
    this->oshape = permute_shape(ishape, dims);
    this->addresses = permute_indices(this->ishape, this->dims);
}

void PermuteDescriptor::resize(int b){
    // Update shapes
    this->ishape[0] = b;
    this->oshape[0] = b;

    // Build indices
    delete addresses;
    this->build_indices();
}

void PermuteDescriptor::build_indices(){
    // Compute index translation (output=>input)
    this->addresses = permute_indices(this->ishape, this->dims);
}
