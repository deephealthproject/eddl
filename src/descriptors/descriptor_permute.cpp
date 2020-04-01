
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/descriptors/tensor_descriptors.h"
#include "eddl/utils.h"

PermuteDescriptor::PermuteDescriptor(const vector<int>& dims, int dev) : SelDescriptor(dev) {
    this->dims = vector<int>(dims);
}


void PermuteDescriptor::build(vector<int> ishape){
    // Get input/output shapes
    this->ishape = ishape;
    this->oshape = permute_shape(ishape, dims);
    this->cpu_addresses = permute_indices(this->ishape, this->dims);
}

void PermuteDescriptor::resize(int b){
    // Delete previous allocations
    this->free_memory();

    // Update shapes
    this->ishape[0] = b;
    this->oshape[0] = b;

    // Build indices
    this->build_indices();

}

void PermuteDescriptor::build_indices(){
    // Compute index translation (output=>input)
    this->cpu_addresses = permute_indices(this->ishape, this->dims);
}
