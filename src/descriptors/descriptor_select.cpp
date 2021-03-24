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

SelDescriptor::SelDescriptor(int dev) : TensorDescriptor(dev) {

}

SelDescriptor::SelDescriptor(const vector<string>& indices, int dev) : TensorDescriptor(dev) {
    this->indices = vector<string>(indices);
}

void SelDescriptor::build(vector<int> ishape){
    // Compute ranges
    this->idxs_range = parse_indices(this->indices, ishape);

    // Get input/output shapes
    this->ishape = ishape;
    this->oshape = indices2shape(this->idxs_range);

    // Build indices
    this->build_indices();
}

void SelDescriptor::resize(int b){

    build_indices();
}

void SelDescriptor::build_indices(){
    // Delete previous allocations
    this->free_memory();

    // Compute index translation (output=>input)
    this->cpu_addresses = ranges2indices(this->ishape, this->idxs_range);
}
