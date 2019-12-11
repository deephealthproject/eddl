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

SelDescriptor::SelDescriptor(const vector<string>& indices){
    this->indices = vector<string>(indices);  // Without batch
}

void SelDescriptor::build(Tensor *A){
    this->input = A;

    // Compute range of indices (add batch)
    vector<string> temp_idxs(this->indices);
    temp_idxs.insert(temp_idxs.begin(), ":");

    // Compute ranges
    this->idxs_range = parse_indices(temp_idxs, this->input->shape);

    // Get input/output shapes
    this->ishape = vector<int>(this->input->shape);
    this->oshape = indices2shape(this->idxs_range);
}

void SelDescriptor::resize(int b){
    // Update batch of range
    this->idxs_range[0][1] = b-1;

    // Update shapes
    this->ishape[0] = b;
    this->oshape[0] = b;

    // Compute index translation (output=>input)
    this->addresses = this->input->ranges2indices(this->idxs_range);
}












////
