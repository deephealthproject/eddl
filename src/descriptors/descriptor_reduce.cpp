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
#include <algorithm>


ReduceDescriptor2::ReduceDescriptor2(const vector<int>& axis, bool keepdims, int dev) : TensorDescriptor(dev) {
    this->axis = axis;
    this->keepdims = keepdims;
}

ReduceDescriptor2::~ReduceDescriptor2(){
    // For FPGA?
}

void ReduceDescriptor2::compute_output(){
    if (this->keepdims){
        this->oshape = vector<int>(this->ishape);
    }else{
        this->oshape = vector<int>();

        // Get output shape: {5, 3, 2} (axis=1) => {5, 2}
        for(int i= 0; i<this->ishape.size(); i++) {

            // Check if axis i (pos) is going to be reduced
            if (find(this->axis.begin(), this->axis.end(), i) == this->axis.end())
                this->oshape.push_back(this->ishape[i]);
        }
    }
}

void ReduceDescriptor2::build_indices() {
    vector<int> istride = shape2stride(this->ishape);

    // indexes
    // get indexes for reduction
    index.clear();

    vector<int> ind;
    ind.push_back(0);
    for(int i=0; i<this->ishape.size(); i++) {
        // Check if "this" dimension is going to be reduced
        bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
        if (!isFound) {  // Dims to not be reduced...
            int s=ind.size();
            for(int j=0;j<s;j++){
                for(int k=0; k<this->ishape[i]-1; k++){
                    ind.push_back(ind[j]+(k+1)*istride[i]);
                }
            }
        }
    }

    sort(ind.begin(), ind.end());

    // reduce through axis to be reduced
    float max,sum;
    int imax;
    for(int i=0;i<ind.size();i++){
        // get axis to be reduced
        index.push_back(vector<int>());

        index[i].push_back(ind[i]);
        for(int l=0;l<this->ishape.size();l++) {
            // Check if "this" dimension is going to be reduced
            bool isFound = find(axis.begin(), axis.end(), l) != axis.end();
            if (isFound) {  // Dims to be reduced...
                int s=index[i].size();
                for(int j=0;j<s;j++){
                    for(int k=0;k<this->ishape[l]-1;k++){
                        index[i].push_back(index[i][j]+(k+1)*istride[l]);
                    }
                }
            }
        }

    }
}

void ReduceDescriptor2::build(const vector<int>& t_ishape){
    this->ishape = vector<int>(t_ishape);

    // Check dims
    if (this->axis.size() >= ishape.size()){
        msg("axis must be lower than tensor dim","ReduceDescriptor");
    }

    // Check axis to reduce
    for(int i=0; i<this->axis.size(); i++){
        if (this->axis[i] >= this->ishape.size()) {
            throw std::runtime_error("axis " + std::to_string(axis[i]-1) + " >= dim=" + std::to_string(this->ishape.size()-1));
        }
    }

    // Compute output dimension
    compute_output();

    // Compute indices to reduce
    build_indices();

    // Compute size reduction
    this->size_reduction = (int)shape2size(this->ishape)/shape2size(this->oshape);
}

void ReduceDescriptor2::resize(int b){
    // Delete previous allocations
    this->free_memory();

    this->ishape[0] = b;
    this->oshape[0] = b;

    this->build(this->ishape);

}

void ReduceDescriptor2::build_map(bool reverse){
    this->free_memory();

    int size = shape2size(this->ishape);
    this->cpu_addresses = new int[size];

    if (!reverse){ // Non-contiguous addresses to reduce.
        #pragma omp parallel for
        for(int i=0; i<index.size(); i++) {  // Reduce index
            for(int j=0; j<index[i].size(); j++){  // Addresses to reduce
                cpu_addresses[index[i][j]] = i;  // A[Original address to reduce] = reduction address
            }
        }
    }else{ // Contiguous addresses to reduce.
        int k=0;
        for(int i=0; i<index.size(); i++) {  // Reduce index
            for(int j=0; j<index[i].size(); j++){  // Addresses to reduce
                cpu_addresses[k++] = index[i][j];  // A[0,1,2...] = [Original address to reduce]
            }
        }
    }
}