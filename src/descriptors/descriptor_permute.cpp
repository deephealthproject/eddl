
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "descriptors.h"
#include "../tensor/tensor_reduction.h"
#include "../utils.h"

PermuteDescriptor::PermuteDescriptor(const vector<int>& dims){
    this->dims = vector<int>(dims);  // Without batch
}

void PermuteDescriptor::build(Tensor *A){
    this->input = A;

    // Get permute dimensions with add batch
    this->dims_batch = {0};
    for(auto &d : this->dims){ dims_batch.emplace_back(d + 1); }

    // Get input/output shapes
    this->ishape = vector<int>(this->input->shape);
    this->oshape = permute_shape(ishape, dims_batch);
}

void PermuteDescriptor::resize(int b){
    // Update shapes
    this->ishape[0] = b;
    this->oshape[0] = b;

    // Compute index translation (output=>input)
    this->addresses = permute_indices(input->shape, this->dims_batch);
}












////
