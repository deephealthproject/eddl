
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <iostream>
#include "eddl/descriptors/tensor_descriptors.h"
#include "eddl/utils.h"

TileDescriptor::TileDescriptor(vector<int> vrepeats, int dev) : SelDescriptor(dev) {
    this->vrepeats = vector<int> (vrepeats);

    // Compute repeats per elements
    this->elem_repeats = 1;
    for(int i=0; i<this->vrepeats.size(); i++){ this->elem_repeats *= vrepeats[i]; }
}


void TileDescriptor::build(vector<int> ishape){
    // Clear shapes
    this->ishape.clear();
    this->oshape.clear();

    // Set input shape
    this->ishape = ishape;

    // Set output shape
    for(int i=0; i<this->ishape.size(); i++){
        oshape.push_back(this->ishape[i] * this->vrepeats[i]);
    }

    // Build indices
    this->build_indices();
}

void TileDescriptor::resize(int b){
    this->build_indices();
}

void TileDescriptor::build_indices(){
    // Get struct data
    int isize = shape2size(this->ishape);
    int osize = shape2size(this->oshape);
    vector<int> A_strides = shape2stride(this->ishape);
    vector<int> B_strides = shape2stride(this->oshape);
    int ndim = this->ishape.size();

    // Delete previous allocations
    this->free_memory();

    // Reserve memory
    this->cpu_addresses = new int[osize];

    // Compute index translation (output=>input)
    for (int A_address=0; A_address<isize; A_address++) {
        vector<int> A_indices(ndim, 0);
        vector<int> B_indices(ndim, 0);

        // Get A Indices
        fast_address2indices(A_address, A_indices.data(), this->ishape.data(),A_strides.data(), ndim);

        vector<vector<int>> axis_idxs;
        for(int axis=0; axis<ndim; axis++) { // 2
            vector<int> tmp_axis_idxs;
            tmp_axis_idxs.reserve(vrepeats[axis]);
            for (int i_rep = 0; i_rep < vrepeats[axis]; i_rep++) { // (2, 2) => 4
                tmp_axis_idxs.push_back(this->ishape[axis]*i_rep + A_indices[axis]);
            }
            axis_idxs.push_back(tmp_axis_idxs);
        }

        vector<vector<int>> tmp_indices = cartesian_product(axis_idxs);
        for(int i=0; i<tmp_indices.size(); i++) {
//            std::cout << "B => (";  for(int k=0; k<ndim; k++) { std::cout << tmp_indices[i][k] << ", "; } std::cout << ")" << std::endl;
            int B_address = fast_indices2address(tmp_indices[i].data(), B_strides.data(), ndim);
            this->cpu_addresses[B_address] = A_address;
        }
    }
}
