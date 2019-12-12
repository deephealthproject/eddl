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

SelDescriptor::SelDescriptor()
{

}
SelDescriptor::~SelDescriptor()
{
  if (addresses==nullptr) delete addresses;
}

SelDescriptor::SelDescriptor(const vector<string>& indices) : TensorDescriptor() {
    addresses=nullptr;
    this->indices = vector<string>(indices);
}

void SelDescriptor::build(vector<int> ishape){
    // Compute ranges
    this->idxs_range = parse_indices(this->indices, ishape);

    // Get input/output shapes
    this->ishape = ishape;
    this->oshape = indices2shape(this->idxs_range);
    this->addresses = ranges2indices(this->ishape, this->idxs_range);
}

void SelDescriptor::resize(int b){
    // Update batch of range
    this->idxs_range[0][1] = b-1;

    // Update shapes
    this->ishape[0] = b;
    this->oshape[0] = b;

    delete addresses;
    build_indices();
}

void SelDescriptor::build_indices(){
    // Compute index translation (output=>input)
    this->addresses = ranges2indices(this->ishape, this->idxs_range);
}
