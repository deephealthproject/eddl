
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/descriptors/tensor_descriptors.h"
#include "eddl/utils.h"

RepeatDescriptor::RepeatDescriptor(vector<unsigned int> vrepeats, unsigned int axis, int dev) : SelDescriptor(dev) {
    this->vrepeats = vector<unsigned int> (vrepeats);
    this->axis = axis;
}


void RepeatDescriptor::build(vector<int> ishape){
    // Clear shapes
    this->ishape.clear();
    this->oshape.clear();

    // Set input shape
    this->ishape = ishape;

    // Set output shape
    for(int i=0; i<ishape.size(); i++){
        unsigned int dsize = 0;
        if(i!=axis){
            dsize = ishape[i];
        }else{
            for(auto &d : this->vrepeats) { dsize+= d; }
        }
        oshape.push_back((int)dsize);
    }

    // Build indices
    this->build_indices();
}

void RepeatDescriptor::resize(int b){
//    // Update shapes
//    this->ishape[0] = b;
//    this->oshape[0] = b;

    // Build indices
    this->build_indices();
}

void RepeatDescriptor::build_indices(){
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
        auto* A_indices = new unsigned int[ndim];
        auto* B_indices = new unsigned int[ndim];

        // Get A Indices
        fast_address2indices(A_address, A_indices, reinterpret_cast<const unsigned int *>(this->ishape.data()),
                             reinterpret_cast<const unsigned int *>(A_strides.data()), ndim);

        // Get B indices. Same as A indices but changing size in axis to be expanded
        std::copy(A_indices, A_indices+ndim, B_indices);  // Copy A indices
        // Get A_indices[axis]=> repeat(3,2,1) AND "sel_index=2" => start at position: 3+2=5
        unsigned int A_idx_axis = A_indices[axis]; // (2, 0) => axis=0 => 2
        unsigned int B_idx_axis = 0;
        for (unsigned int j = 0; j < A_idx_axis; j++) { B_idx_axis+= this->vrepeats[j]; }
        B_indices[axis] = B_idx_axis;

        // Copy value t times
        unsigned int B_address = fast_indices2address(B_indices, reinterpret_cast<const unsigned int *>(B_strides.data()), ndim);
        for (unsigned int t = 0; t < vrepeats[A_indices[axis]]; t++) {
            this->cpu_addresses[B_address + t*B_strides[axis]] = A_address;
        }

        // Delete stuff
        delete[] A_indices;
        delete[] B_indices;
    }
}
