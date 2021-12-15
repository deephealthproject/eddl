/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include <utility>

#include "eddl/tensor/tensor.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

using namespace std;

// ***** Core (in-place) *****************************
void Tensor::fill_(float v) {
    Tensor::fill(this, v);
}

Tensor* Tensor::fill(float v){
    Tensor* t_new = Tensor::empty_like(this);
    Tensor::fill(t_new, v);
    return t_new;
}

void Tensor::fill(Tensor* A, float v){
    if (A->isCPU()) {
        cpu_fill_(A, v);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_fill_(A, v);
      }
#endif
}


void Tensor::permute_(const vector<int>& dims){
    Tensor* temp = Tensor::permute(this, dims);

    // Update attributes
    updateShape(temp->shape);
    updateSize();
    updateStrides();
    Tensor::copy(temp, this);  // copy data

    delete temp;
}


Tensor* Tensor::permute(const vector<int>& dims){
    Tensor* t_new = Tensor::permute(this, dims);
    return t_new;
}

Tensor* Tensor::permute(Tensor* A, const vector<int>& dims){
    // Build descriptor
    auto *sd = new PermuteDescriptor(dims, A->device);
    sd->build(A->shape);

    // Initialize new tensor
    auto *new_t = new Tensor(sd->oshape, A->device);

    // Fill new tensor
    Tensor::select(A, new_t, sd);

    delete sd;
    return new_t;
}


void Tensor::moveaxis_(int source, int destination){
    Tensor* temp = Tensor::moveaxis(this, source, destination);

    // Update attributes
    updateShape(temp->shape);
    updateSize();
    updateStrides();
    Tensor::copy(temp, this);  // copy data

    delete temp;
}

Tensor* Tensor::moveaxis(int source, int destination){
    Tensor* t_new = Tensor::moveaxis(this, source, destination);
    return t_new;
}

Tensor* Tensor::moveaxis(Tensor* A, int source, int destination){
    // Check values
    if(source<-1 || destination <-1){
        msg("Invalid axis", "Tensor::moveaxis");
    }

    // User "-1" as alias for the last dimension
    if(source == -1){source = A->ndim-1; }
    if(destination == -1){destination = A->ndim-1; }

    // Build axes to permute [1 => 3] => (0,1,2,3) => (0,2,3,1)
    vector<int> dims;
    dims.reserve(A->ndim);
    for(int i=0; i<A->ndim;i++){
        dims.push_back(i);
    }
    dims.erase(dims.begin()+source);  // Remove axis
    dims.insert(dims.begin() + destination, source);  // Insert at final position

    // Permute tensor
    Tensor* t2 = Tensor::permute(A, dims);
    return t2;
}


void Tensor::swapaxis_(int axis1, int axis2){
    Tensor* temp = Tensor::swapaxis(this, axis1, axis2);

    // Update attributes
    updateShape(temp->shape);
    updateSize();
    updateStrides();
    Tensor::copy(temp, this);  // copy data

    delete temp;
}


Tensor* Tensor::swapaxis(int axis1, int axis2){
    Tensor* t_new = Tensor::swapaxis(this, axis1, axis2);
    return t_new;
}

Tensor* Tensor::swapaxis(Tensor* A, int axis1, int axis2){
    // Check values
    if(axis1<-1 || axis2 <-1 || axis1 == axis2){
        msg("Invalid axis", "Tensor::swapaxis");
    }

    // Build axes to permute [0, 3] => (0,1,2,3) => (3,1,2,0)
    vector<int> dims;
    for(int i=0; i<A->ndim;i++){ dims.emplace_back(i); }
    dims[axis1] = axis2;
    dims[axis2] = axis1;

    // Permute tensor
    Tensor* t2 = Tensor::permute(A, dims);
    return t2;
}


void Tensor::reshape_(const vector<int> &new_shape){
    int new_size = 1;  // For checking
    vector<int> final_shape;

    // Compute new shape (infer if necessary)
    for(auto d : new_shape) {
        if(d==-1){  // Infer the remaining dimensions
            d = this->size/new_size;
        }
        final_shape.push_back(d);
        new_size *= d;
    }

    // Check if the new size is compatible
    if(new_size!=this->size){
        cout<<new_size<<"!="<<size<<endl;
        msg("Not compatible shapes", "Tensor::reshape_");
    }

    // Update attributes
    updateShape(final_shape);
    updateSize();
    updateStrides();
    updateData(this->ptr, nullptr, isshared);  // Due to potential Eigen mapping when CPU and dim=2
}

Tensor* Tensor::reshape(const vector<int> &new_shape){
    Tensor *t_new = Tensor::reshape(this, new_shape);
    return t_new;
}

Tensor* Tensor::reshape(Tensor *A, const vector<int> &shape){
    Tensor *t_new = A->clone();
    t_new->reshape_(shape);
    return t_new;
}


void Tensor::flatten_(){
    this->reshape_({-1});
}

Tensor* Tensor::flatten(){
    Tensor *t_new = Tensor::flatten(this);
    return t_new;
}

Tensor* Tensor::flatten(Tensor *A){
    Tensor *t_new = A->clone();
    t_new->reshape_({-1});
    return t_new;
};


void Tensor::squeeze_(int axis){
    // Remove dimension/s
    vector<int> new_shape = compute_squeeze(this->shape, axis);
    this->reshape_(new_shape);
}

Tensor* Tensor::squeeze(int axis){
    Tensor *t_new = Tensor::squeeze(this, axis);
    return t_new;
}

Tensor* Tensor::squeeze(Tensor *A, int axis){
    Tensor *t_new = A->clone();
    t_new->squeeze_(axis);
    return t_new;
}


void Tensor::unsqueeze_(int axis){
    // Add dimension
    vector<int> new_shape = compute_unsqueeze(this->shape, axis);
    this->reshape_(new_shape);
}

Tensor* Tensor::unsqueeze(int axis){
    Tensor *t_new = Tensor::unsqueeze(this, axis);
    return t_new;
}

Tensor* Tensor::unsqueeze(Tensor *A, int axis){
    Tensor *t_new = A->clone();
    t_new->unsqueeze_(axis);
    return t_new;
}


//int Tensor::get_address_rowmajor(vector<int> indices){
//    int address=0;
//    for(int i=0; i<this->ndim; i++){ address +=  indices[i] * this->stride[i];}  //*(indices.begin()+i)
//    return address;
//}
//
//vector<int> Tensor::get_indices_rowmajor(int address){
//    vector<int> indices;
//    indices.reserve(this->shape.size());
//    for(int i=0; i<this->shape.size(); i++){
//        indices.push_back(address / this->stride[i] % this->shape[i]);
//    }
//    return indices;
//}
//
//float Tensor::get_(vector<int> indices){
//    // DO NOT USE. They're mainly for debugging.
//    return this->ptr[get_address_rowmajor(std::move(indices))];
//}
//
//void Tensor::set_(vector<int> indices, float value){
//    // DO NOT USE. They're mainly for debugging.
//    this->ptr[get_address_rowmajor(std::move(indices))] = value;
//}
//
//bool Tensor::valid_indices(vector<int> indices){
//    for (int i=0; i<indices.size(); i++){
//        if (indices[i] <0 || indices[i] >= this->shape[i]){
//            return false;
//        }
//    }
//    return true;
//}

// ***** Core (static) *****************************
void Tensor::transpose(Tensor *A, Tensor *B, vector<int> dims) {
    // TODO: Deprecated.
    // Transpose

    if (A->size != B->size)
        msg("Tensors with different size", "Tensor::transpose");

    if (A->device != B->device) msg("Tensors in different devices", "Tensor::transpose");

    Tensor *N;
    if (A == B) N = new Tensor(A->getShape(), A->device);
    else N = B;


    // Copy tensor data
    if (A->isCPU()) {
        cpu_transpose(A, N);
    }
#ifdef cGPU
    else if (A->isGPU())
      {

      }
#endif

    if (A == B) delete N;

}

void Tensor::copy(Tensor *A, Tensor *B) {
    ///////////////////////////////////////
    /// Copy from A to B
    //////////////////////////////////////

    if (!Tensor::sameSize(A, B)) {
        A->info();
        B->info();
        msg("Tensors with different size", "Tensor::copy");
    }


    if ((A->isCPU()) && (B->isCPU())) {
        cpu_copy(A, B);
    }
#ifdef cGPU
        else if ((A->isGPU())&&(B->isGPU())) {
          gpu_copy_gpu(A,B);
        }
        else if ((A->isCPU())&&(B->isGPU()))
          {
            gpu_copy_to_gpu(A->ptr,B);
          }
        else if ((A->isGPU())&&(B->isCPU()))
          {
            gpu_copy_from_gpu(A,B->ptr);
          }
#endif
    else {
        fprintf(stderr, "(%d %d)\n", A->device, B->device);
        msg("unsupported copy between devices", "Tensor::copy");
    }

}


void Tensor::fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc) {
    ///////////////////////////////////////
    /// Partial copy ndim=1
    //////////////////////////////////////
    if (A->ndim != B->ndim)
        msg("Tensors with different shape", "Tensor::fill");


    if ((A->isCPU()) && (B->isCPU())) {
        cpu_fill(A, aini, aend, B, bini, bend, inc);
    }
#ifdef cGPU
        else if ((A->isGPU())&&(B->isGPU())) {
          gpu_fill(A,aini,aend,B,bini,bend,inc);
        }
#endif
    else {
        fprintf(stderr, "(%d %d)\n", A->device, B->device);
        msg("unsupported copy between devices", "Tensor::copy");
    }

}

void Tensor::sort_(bool descending, bool stable){
    Tensor::sort(this, this, descending, stable);
}

Tensor* Tensor::sort(bool descending, bool stable){
    Tensor *t = Tensor::empty_like(this);
    Tensor::sort(this, t, descending, stable);
    return t;
}

void Tensor::sort(Tensor* A, Tensor* B, bool descending, bool stable){
    if (A->isCPU() && B->isCPU()){
        cpu_sort(A, B, descending, stable);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_sort(A, B, descending, stable);
    }
#endif
}



Tensor* Tensor::argsort(bool descending, bool stable){
    Tensor *t = Tensor::empty_like(this);
    Tensor::argsort(this, t, descending, stable);
    return t;
}

void Tensor::argsort(Tensor* A, Tensor* B, bool descending, bool stable){
    if (A->isCPU() && B->isCPU()){
        cpu_argsort(A, B, descending, stable);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_argsort(A, B, descending, stable);
    }
#endif
}

Tensor* Tensor::concat(const vector<Tensor*> A, unsigned int axis, Tensor* output){
    // Check number of vectors to concat
    if(A.size()<2){
        msg("Concat requires a minimum of two tensors", "Tensor::concat");
    }

    // Temp variables
    vector<int> new_shape = A[0]->shape;
    int new_axis = 0;

    // Walk through each tensor to check for compatibility issues (from 1 to n)
    for(int i=1; i<A.size(); i++){

        // Check device
        if(A[0]->device != A[i]->device){
            msg("All tensors must be on the same device", "Tensor::concat");
        }

        // Check dimensions
        if(A[0]->ndim != A[i]->ndim){
            msg("The number of dimensions of all tensors must match (" +
                to_string(A[0]->ndim) +  "!=" + to_string(A[i]->ndim) + ")", "Tensor::concat");
        }


        // Check that all dimensions match except the one to concat
        for(int j=0; j<A[0]->shape.size(); j++) {

            // Check current dimension
            if (j!=axis && A[0]->shape[j] != A[i]->shape[j]) {
                msg("The dimensions across of all tensors must match (" +
                    to_string(A[0]->shape[j]) +  "!=" + to_string(A[i]->shape[j]) + ")", "Tensor::concat");
            }
        }

        // Sum dimension
        new_axis += A[i]->shape[axis];
    }

    // Update final shape
    new_shape[axis] +=  new_axis; // new_shape[axis] had the shape of the first tensor

    // Create new tensor
    if(output==nullptr){
        output = new Tensor(new_shape, A[0]->device);
    }else{
        // Check dimensions
        if(output->shape!=new_shape){
            msg("The dimension of the output tensor is incorrect", "Tensor::concat");
        }else if(output->device != A[0]->device){
            msg("The output tensor and the input ones must be on the same device", "Tensor::concat");
        }
    }

    if (output->isCPU()) {
        cpu_concat(output, A, axis, false);
    }
#ifdef cGPU
    else if (output->isGPU())
      {
        gpu_concat(output, A, axis, false);
      }
#endif
    return output;
}

void Tensor::concat_back(Tensor *A, const vector<Tensor*> t, unsigned int axis){
    // Check number of vectors to concat
    if(t.size()<2){
        msg("Concat back requires a minimum of two tensors", "Tensor::concat_back");
    }

    // Temp variables
    vector<int> new_shape = t[0]->shape;
    int new_axis = 0;

    // Walk through each tensor to check for compatibility issues (from 1 to n)
    for(int i=1; i<t.size(); i++){

        // Check device
        if(t[0]->device != t[i]->device){
            msg("All tensors must be on the same device", "Tensor::concat_back");
        }

        // Check dimensions
        if(t[0]->ndim != t[i]->ndim){
            msg("The number of dimensions of all tensors must match (" +
                to_string(t[0]->ndim) +  "!=" + to_string(t[i]->ndim) + ")", "Tensor::concat_back");
        }


        // Check that all dimensions match except the one to concat
        for(int j=0; j<t[0]->shape.size(); j++) {

            // Check current dimension
            if (j!=axis && t[0]->shape[j] != t[i]->shape[j]) {
                msg("The dimensions across of all tensors must match (" +
                    to_string(t[0]->shape[j]) +  "!=" + to_string(t[i]->shape[j]) + ")", "Tensor::concat_back");
            }
        }

        // Sum dimension
        new_axis += t[i]->shape[axis];
    }

    // Check input shape
    new_shape[axis] +=  new_axis;
    if(new_shape!=A->shape){
        msg("Mismatched shape between output tensor and input tensors", "Tensor::concat_back");
    }

    // Check device (again)
    if(A->device != t[0]->device){
        msg("All tensors must be on the same device", "Tensor::concat_back");
    }

    // Perform operations
    if (A->isCPU()) {
        cpu_concat(A, t, axis, true);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_concat(A, t, axis, true);
      }
#endif
}

Tensor* Tensor::stack(const vector<Tensor*> A, unsigned int axis, Tensor* output){
    // Check number of vectors to concat
    if(A.size()<2){
        msg("Stack requires a minimum of two tensors", "Tensor::stack");
    }

    // Create fake tensors with dimension expanded
    vector<Tensor*> tmp_tensors;
    for(int i=0; i<A.size(); i++) {
        // Create fake tensor (ptr reference)
        Tensor* tmp = new Tensor(A[i]->shape, A[i]->ptr, A[i]->device);

        // Expand dimension
        tmp->unsqueeze_(axis);

        // Add tmp tensor to buffer
        tmp_tensors.push_back(tmp);
    }

    vector<int> new_shape = tmp_tensors[0]->shape;
    new_shape[axis] = tmp_tensors.size();

    // Create new tensor
    if(output==nullptr){
        output = new Tensor(new_shape, A[0]->device);
    }else{
        // Check dimensions
        if(output->shape!=new_shape){
            msg("The dimension of the output tensor is incorrect", "Tensor::stack");
        }else if(output->device != A[0]->device){
            msg("The output tensor and the input ones must be on the same device", "Tensor::stack");
        }
    }

    // Concat tensors in the expanded axis
    Tensor::concat(tmp_tensors, axis, output);

    // Delete fake tensors
    for(int i=0; i<A.size(); i++) {
        tmp_tensors[i]->ptr = nullptr; // Dereference tensor
        delete tmp_tensors[i]; tmp_tensors[i] = nullptr;
    }
    tmp_tensors.clear();


    return output;
}

Tensor* Tensor::repeat(Tensor* A, const vector<unsigned int>& repeats, unsigned int axis, Tensor* output, bool derivative){
    // NOT USED ANYWHERE. REPEAT DONE THROUGH A SELECT
    // Check axis values
    if(axis<0 || axis > A->ndim-1){
        msg("The axis must be a number between 0 and the maximum dimension of the tensor", "Tensor::repeat");
    }

    // Check that there are enough values in
    if(repeats.size()!=A->shape[axis]){
        msg("The size of 'repeats' (" + std::to_string(repeats.size()) + ") must equal the size the the dimension to repeat " + std::to_string(A->shape[axis]) + ")", "Tensor::repeat");
    }

    // Compute new shape
    vector<int> new_shape;
    for(int i=0; i<A->ndim; i++){
        unsigned int dsize = 0;
        if(i!=axis){
            dsize = A->shape[i];
        }else{
            for(auto &d : repeats) { dsize+= d; }
        }
        new_shape.push_back((int)dsize);
    }

    // Create new tensor
    if(output==nullptr){
        output = new Tensor(new_shape, A->device);
    }else{
        // Check dimensions
        if(output->shape!=new_shape){
            msg("The dimension of the output tensor is incorrect", "Tensor::repeat");
        }else if(output->device != A->device){
            msg("The output tensor and the input ones must be on the same device", "Tensor::repeat");
        }
    }

    if (A->isCPU() && output->isCPU()){
        cpu_repeat(A, output, repeats, axis, derivative);
    }
#ifdef cGPU
    else if (A->isGPU() && output->isGPU())
    {
        gpu_repeat(A, output, repeats, axis, derivative);
    }
#endif
    return output;
}

Tensor* Tensor::repeat(Tensor* A, unsigned int repeats, unsigned int axis, Tensor* output, bool derivative){
    // Check axis values
    if(axis<0 || axis > A->ndim-1){
        msg("The axis must be a number between 0 and the maximum dimension of the tensor", "Tensor::repeat");
    }
    // Repeat n times each dimension
    vector<unsigned int> vrepeats = vector<unsigned int>(A->shape[axis], repeats);

    // Call main function
    return Tensor::repeat(A, vrepeats, axis, output, derivative);
}

Tensor* Tensor::repeat_desc(Tensor* A, const vector<unsigned int>& repeats, unsigned int axis, Tensor* output){
    // Build descriptor
    auto *rd = new RepeatDescriptor(repeats, axis, A->device);
    rd->build(A->shape);

    // Create new tensor
    if(output==nullptr){
        output = new Tensor(rd->oshape, A->device);
    }else{
        // Check dimensions
        if(output->shape!=rd->oshape){
            msg("The dimension of the output tensor is incorrect", "Tensor::repeat_desc");
        }else if(output->device != A->device){
            msg("The output tensor and the input ones must be on the same device", "Tensor::repeat_desc");
        }
    }

    // Fill new tensor
    Tensor::select(A, output, rd);

    delete rd;
    return output;
}


Tensor* Tensor::repeat_desc(Tensor* A, unsigned int repeats, unsigned int axis, Tensor* output){
    // Check axis values
    if(axis<0 || axis > A->ndim-1){
        msg("The axis must be a number between 0 and the maximum dimension of the tensor", "Tensor::repeat_desc");
    }
    // Repeat n times each dimension
    vector<unsigned int> vrepeats = vector<unsigned int>(A->shape[axis], repeats);

    // Call main function
    return Tensor::repeat_desc(A, vrepeats, axis, output);
}

Tensor* Tensor::tile(Tensor* A, const vector<int>& repeats){
    // Check dimensions
    if(A->ndim != repeats.size()){
        msg("The number of dimensions in tensor 'A' must match the size of 'repeats'", "Tensor::tile");
    }

    // Dimensions must be positive
    for(int i=0; i<repeats.size(); i++){
        if(repeats[i] < 1){
            msg("All repetitions must be greater or equal than 1", "Tensor::tile");
        }
    }

    // Build descriptor
    auto *td = new TileDescriptor(repeats, A->device);
    td->build(A->shape);

    // Initialize tensor
    auto* new_t = new Tensor(td->oshape, A->device);

    // Perform select
    Tensor::select(A, new_t, td);

    delete td;
    return new_t;
}

Tensor* Tensor::select(const vector<string>& indices){
    // Build descriptor
    auto *sd = new SelDescriptor(indices, this->device);
    sd->build(this->shape);

    // Initialize tensor
    auto* t = new Tensor(sd->oshape, this->device);

    // Perform select
    Tensor::select(this, t, sd);

    delete sd;
    return t;
}

void Tensor::select(Tensor *A, Tensor* B, SelDescriptor *sd){
    if (A->isCPU() && B->isCPU()) {
        cpu_select(A, B, sd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_select(A, B, sd);
      }
#endif
}

void Tensor::select_back(Tensor *A, Tensor* B, SelDescriptor *sd){
    if (A->isCPU() && B->isCPU()) {
        cpu_select_back(A, B, sd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_select_back(A, B, sd);
      }
#endif
}

void Tensor::set_select(const vector<string>& indices, float value){
    auto *sd = new SelDescriptor(indices, this->device);
    sd->build(this->shape);

    Tensor* A = Tensor::full(sd->oshape, value);

    // Check if the dimensions of the selection and the tensor are compatibles
    if(sd->oshape==A->shape){
        Tensor::set_select(this, A, sd);
    }else{
      
        msg("Incompatible dimensions", "Tensor::set_select");
    }

    delete A;
    delete sd;
}

void Tensor::set_select(const vector<string>& indices, Tensor *A){
    auto *sd = new SelDescriptor(indices, this->device);
    sd->build(this->shape);

    // Check if the dimensions of the selection and the tensor are compatibles
    if(sd->oshape==A->shape){
        Tensor::set_select(this, A, sd);
    }else{
        info();
        A->info();

        msg("Incompatible dimensions", "Tensor::set_select");
    }

    delete sd;
}

void Tensor::set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    if (A->isCPU() && B->isCPU()) {
        cpu_set_select(A, B, sd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_set_select(A, B, sd);
      }
#endif
}


void Tensor::set_select_back(Tensor *A, Tensor* B, SelDescriptor *sd){
    if (A->isCPU() && B->isCPU()) {
        cpu_set_select_back(A, B, sd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
      {
        gpu_set_select_back(A, B, sd);
      }
#endif

}

void Tensor::gather(Tensor *A, Tensor *B, GatherDescriptor *sd){
    if (A->isCPU() && B->isCPU()) {
        cpu_gather(A, B, sd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_gather(A, B, sd);
    }
#endif
}

Tensor* Tensor::expand(int size){
    return Tensor::expand(this, size);
}


Tensor* Tensor::expand(Tensor *A, int size){
    // Build descriptor
    auto *sd = new ExpandDescriptor(size, A->device);
    sd->build(A->shape);

    // Initialize tensor
    auto* t = new Tensor(sd->oshape, A->device);

    // Perform select
    Tensor::expand(A, t, sd);

    delete sd;
    return t;
}

void Tensor::expand(Tensor *A, Tensor *B, ExpandDescriptor *sd){
    if (A->isCPU() && B->isCPU()) {
        cpu_expand(A, B, sd);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_expand(A, B, sd);
    }
#endif
}

void Tensor::select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end, bool mask_zeros) {
    ///////////////////////////////////////
    /// Select from A to B, A is bigger
    //////////////////////////////////////

    if ((A->size / A->shape[0]) != (B->size / B->shape[0])) {
        A->info();
        B->info();
        msg("Incompatible shape", "Tensor::select");
    }



    if ((A->isCPU()) && (B->isCPU())) {
        cpu_select(A, B, sind, ini, end,mask_zeros);
    }
    else if ((A->isGPU()) && (B->isCPU())) {
        Tensor *Ac=A->clone();
        Ac->toCPU();

        cpu_select(Ac, B, sind, ini, end,mask_zeros);

        delete Ac;
    }else if ((A->isCPU()) && (B->isGPU())) {

        Tensor *Bc=B->clone();
        Bc->toCPU();

        cpu_select(A, Bc, sind, ini, end,mask_zeros);

        Tensor::copy(Bc,B);

        delete Bc;

    }
    else if ((A->isFPGA()) && (B->isCPU())) {
        Tensor *Ac=A->clone();
        Ac->toCPU();

        cpu_select(Ac, B, sind, ini, end,mask_zeros);

        delete Ac;
    }else if ((A->isCPU()) && (B->isFPGA())) {
        Tensor *Bc=B->clone();
        Bc->toCPU();
        cpu_select(A, Bc, sind, ini, end,mask_zeros);

        Tensor::copy(Bc,B);

        delete Bc;
    }
    #ifdef cGPU
        else if (A->isGPU() && B->isGPU())
      {
        gpu_select(A, B, sind, ini, end,mask_zeros);
      }
    #endif

    else {
        msg("unsuppoted select", "Tensor::select");
    }

}
void Tensor::deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,int inc, bool mask_zeros) {
    ///////////////////////////////////////
    /// deSelect from A to B, B is bigger
    //////////////////////////////////////

    if ((A->size / A->shape[0]) != (B->size / B->shape[0])) {
        A->info();
        B->info();
        msg("Incompatible shape", "Tensor::select");
    }


    if ((A->isCPU()) && (B->isCPU())) {
        cpu_deselect(A, B, sind, ini, end, inc,mask_zeros);
    }
    else if ((A->isGPU()) && (B->isCPU())) {
        Tensor *Ac=A->clone();
        Ac->toCPU();

        cpu_deselect(Ac, B, sind, ini, end, inc,mask_zeros);

        delete Ac;
    }else if ((A->isCPU()) && (B->isGPU())) {
        Tensor *Bc=B->clone();
        Bc->toCPU();
        cpu_deselect(A, Bc, sind, ini, end, inc,mask_zeros);

        Tensor::copy(Bc,B);

        delete Bc;
    } else if ((A->isFPGA()) && (B->isCPU())) {
        Tensor *Ac=A->clone();
        Ac->toCPU();

        cpu_deselect(Ac, B, sind, ini, end, inc,mask_zeros);

        delete Ac;
    }else if ((A->isCPU()) && (B->isFPGA())) {
        Tensor *Bc=B->clone();
        Bc->toCPU();
        cpu_deselect(A, Bc, sind, ini, end, inc,mask_zeros);

        Tensor::copy(Bc,B);

        delete Bc;
    }

    #ifdef cGPU
        else if (A->isGPU() && B->isGPU())
      {
        gpu_deselect(A, B, sind, ini, end, inc,mask_zeros);
      }
    #endif

    else {
        msg("unsuppoted select", "Tensor::select");
    }

}

void Tensor::tile_deprecated(Tensor *A, Tensor *B)
{

    int Asize=A->shape[0];
    int Bsize=B->shape[0];


    if (Bsize>Asize) {
        vector<int> sind(Bsize);
        int start,end;
        for(int i=0;i<Bsize;i++) sind[i]=i;
        for(int i=0;i<Bsize/Asize;i++) {
            start = i * Asize;
            end = start + Asize;
            Tensor::deselect(A, B, sind, start, end);
        }
        if (Bsize%Asize) {
            Tensor::deselect(A, B, sind, end, end+(Bsize%Asize));
        }
    }
    else {
        vector<int> sind(Bsize);
        for(int i=0;i<Bsize;i++) sind[i]=i;
        Tensor::select(A, B, sind, 0, Bsize);
    }
}

Tensor* Tensor::broadcast(Tensor* A, Tensor* B, Tensor *output){
    // Source: https://numpy.org/doc/stable/user/basics.broadcasting.html

    vector<int> shape1 = A->shape;
    vector<int> shape2 = B->shape;
    bool shapes_swapped = false;

    // Shape1 must be smaller (or equal) than Shape2
    if(shape1.size()>shape2.size()){ shape1.swap(shape2); shapes_swapped = true; }

    // Get shape to broadcast (normalized)
    vector<int> broadcast_from = getBroadcastShape(shape1, shape2);
    if(broadcast_from.empty()){
        msg("The dimensions of both tensors must be equal or compatible (i.e (3)*(1,3), (3)*(6,2,5,3,5), (5, 3)*(5, 3),...)", "Tensor::broadcast");
    }

    // Get repetitions to perform a given broadcast
    vector<int> tile_repetitions = getTilesRepetitions(broadcast_from, shape2);
    if(tile_repetitions.empty()){
        msg("These tensors cannot be broadcasted. Two dimensions are compatible when: 1) they are equal, or 2) one of them is 1", "Tensor::broadcast");
    }

    // Build descriptor
    auto *td = new TileDescriptor(tile_repetitions, A->device);
    td->build(broadcast_from);

    // Create new tensor (if needed)
    Tensor* new_t;
    if (output== nullptr){
        new_t = new Tensor(td->oshape, A->device);
    }else{
        new_t = output;
    }

    // Do broadcast (tile repetitions)
    if(!shapes_swapped){
        Tensor::select(A, new_t, td);
    }else{
        Tensor::select(B, new_t, td);
    }

    delete td;
    return new_t;
}
