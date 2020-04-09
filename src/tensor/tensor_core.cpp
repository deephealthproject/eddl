/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include <utility>

#include "tensor/tensor.h"
#include "hardware/cpu/cpu_hw.h"

#ifdef cGPU
#include "hardware/gpu/gpu_tensor.h"
#include "hardware/gpu/gpu_hw.h"
#include "hardware/gpu/nn/gpu_nn.h"
#endif

using namespace std;

// ***** Core (in-place) *****************************
void Tensor::fill_(float v) {
    if (this->isCPU()) {
        cpu_fill_(this, v);
    }
#ifdef cGPU
    else if (this->isGPU())
      {
        gpu_fill_(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

//Tensor* Tensor::fill(Tensor *A, float v){
//    Tensor *t_new = A->clone();
//    t_new->fill_(v);
//    return t_new;
//}

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
        msg("Not compatible shapes", "Tensor::reshape_");
    }

    // Update attributes
    updateShape(final_shape);
    updateSize();
    updateStrides();
    updateData(this->ptr);  // Due to the Eigen mapping
}

Tensor* Tensor::reshape(Tensor *A, const vector<int> &shape){
    Tensor *t_new = A->clone();
    t_new->reshape_(shape);
    return t_new;
}

Tensor* Tensor::flatten(Tensor *A){
    Tensor *t_new = A->clone();
    t_new->reshape_({-1});
    return t_new;
};


void Tensor::squeeze_(){
    // Remove single dimension entries from the array
    vector<int> new_shape;
    for(auto &d : this->shape){
        if(d>1){
            new_shape.push_back(d);
        }
    }
    this->reshape_(new_shape);
}

Tensor* Tensor::squeeze(Tensor *A){
    Tensor *t_new = A->clone();
    t_new->squeeze_();
    return t_new;
}

void Tensor::unsqueeze_(){
    vector<int> new_shape(this->shape);
    new_shape.insert(new_shape.begin(), 1); // Add one dimension to the beginning
    this->reshape_(new_shape);
}

Tensor* Tensor::unsqueeze(Tensor *A){
    Tensor *t_new = A->clone();
    t_new->unsqueeze_();
    return t_new;
}

Tensor* Tensor::permute(Tensor* t, const vector<int>& dims){
    // Build descriptor
    auto *sd = new PermuteDescriptor(dims, t->device);
    sd->build(t->shape);
    sd->build_indices();

    // Initialize new tensor
    auto *new_t = new Tensor(sd->oshape, t->device);

    // Fill new tensor
    Tensor::select(t, new_t, sd);
    return new_t;
}

Tensor* Tensor::moveaxis(Tensor* t, int source, int destination){
    // Check values
    if(source<-1 || destination <-1){
        msg("Invalid axis", "Tensor::moveaxis");
    }

    // User "-1" as alias for the last dimension
    if(source == -1){source = t->ndim-1; }
    if(destination == -1){destination = t->ndim-1; }

    // Build axes to permute [1 => 3] => (0,1,2,3) => (0,2,3,1)
    vector<int> dims;
    dims.reserve(t->ndim);
    for(int i=0; i<t->ndim;i++){
        dims.push_back(i);
    }
    dims.erase(dims.begin()+source);  // Remove axis
    dims.insert(dims.begin() + destination, source);  // Insert at final position

    // Permute tensor
    Tensor* t2 = Tensor::permute(t, dims);
    return t2;
}

Tensor* Tensor::swapaxis(Tensor* t, int axis1, int axis2){
    // Check values
    if(axis1<-1 || axis2 <-1 || axis1 == axis2){
        msg("Invalid axis", "Tensor::swapaxis");
    }

    // Build axes to permute [0, 3] => (0,1,2,3) => (3,1,2,0)
    vector<int> dims;
    for(int i=0; i<t->ndim;i++){ dims.emplace_back(i); }
    dims[axis1] = axis2;
    dims[axis2] = axis1;

    // Permute tensor
    Tensor* t2 = Tensor::permute(t, dims);
    return t2;
}


int Tensor::get_address_rowmajor(vector<int> indices){
    int address=0;
    for(int i=0; i<this->ndim; i++){ address +=  indices[i] * this->stride[i];}  //*(indices.begin()+i)
    return address;
}

vector<int> Tensor::get_indices_rowmajor(int address){
    vector<int> indices;
    indices.reserve(this->shape.size());
    for(int i=0; i<this->shape.size(); i++){
        indices.push_back(address / this->stride[i] % this->shape[i]);
    }
    return indices;
}

float Tensor::get_(vector<int> indices){
    return this->ptr[get_address_rowmajor(std::move(indices))];
}

void Tensor::set_(vector<int> indices, float value){
    this->ptr[get_address_rowmajor(std::move(indices))] = value;
}

bool Tensor::valid_indices(vector<int> indices){
    for (int i=0; i<indices.size(); i++){
        if (indices[i] <0 || indices[i] >= this->shape[i]){
            return false;
        }
    }
    return true;
}
// ***** Core (static) *****************************
void Tensor::transpose(Tensor *A, Tensor *B, vector<int> dims) {
    // Transpose
    // TODO: Review correctness
    B->tsem->lock();
    if (!Tensor::eqsize(A, B))
        msg("Tensors with different shape", "Tensor::transpose");

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
#ifdef cFPGA
    else {

    }
#endif
    B->tsem->unlock();

    if (A == B) delete N;

}

void Tensor::copy(Tensor *A, Tensor *B) {
    ///////////////////////////////////////
    /// Copy from A to B
    //////////////////////////////////////
    // TODO: Review correctness for ndim==2

    if (!Tensor::eqsize(A, B)) {
        A->info();
        B->info();
        msg("Tensors with different shape", "Tensor::copy");
    }

    B->tsem->lock();
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
    B->tsem->unlock();
}


void Tensor::fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc) {
    ///////////////////////////////////////
    /// Partial copy ndim=1
    //////////////////////////////////////
    if (A->ndim != B->ndim)
        msg("Tensors with different shape", "Tensor::fill");

    B->tsem->lock();
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
    B->tsem->unlock();
}

Tensor* Tensor::concat(const vector<Tensor*> t, unsigned int axis, Tensor* output){
    // Check number of vectors to concat
    if(t.size()<2){
        msg("Concat requires a minimum of two tensors", "Tensor::concat");
    }

    // Temp variables
    vector<int> new_shape = t[0]->shape;
    int new_axis = 0;

    // Walk through each tensor to check for compatibility issues (from 1 to n)
    for(int i=1; i<t.size(); i++){

        // Check device
        if(t[0]->device != t[i]->device){
            msg("All tensors must be on the same device", "Tensor::concat");
        }

        // Check dimensions
        if(t[0]->ndim != t[i]->ndim){
            msg("The number of dimensions of all tensors must match (" +
                to_string(t[0]->ndim) +  "!=" + to_string(t[i]->ndim) + ")", "Tensor::concat");
        }


        // Check that all dimensions match except the one to concat
        for(int j=0; j<t[0]->shape.size(); j++) {

            // Check current dimension
            if (j!=axis && t[0]->shape[j] != t[i]->shape[j]) {
                msg("The dimensions across of all tensors must match (" +
                    to_string(t[0]->shape[j]) +  "!=" + to_string(t[i]->shape[j]) + ")", "Tensor::concat");
            }
        }

        // Sum dimension
        new_axis += t[i]->shape[axis];
    }

    // Update final shape
    new_shape[axis] +=  new_axis; // new_shape[axis] had the shape of the first tensor

    // Create new tensor
    if(output==nullptr){
        output = new Tensor(new_shape, t[0]->device);
    }else{
        // Check dimensions
        if(output->shape!=new_shape){
            msg("The dimension of the output tensor is incorrect", "Tensor::concat");
        }else if(output->device != t[0]->device){
            msg("The output tensor and the input ones must be on the same device", "Tensor::concat");
        }
    }

    if (output->isCPU()) {
        cpu_concat(output, t, axis, false);
    }
#ifdef cGPU
    else if (output->isGPU())
      {
        gpu_concat(output, t, axis, false);
      }
#endif
#ifdef cFPGA
    else {

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
#ifdef cFPGA
    else {

    }
#endif
}


Tensor* Tensor::select(const vector<string>& indices){
    Tensor* t = nullptr;

    auto *sd = new SelDescriptor(indices, this->device);
    sd->build(this->shape);
    sd->build_indices();

    // Initialize tensor
    t = new Tensor(sd->oshape, this->device);

    // Perform select
    Tensor::select(this, t, sd);
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
#ifdef cFPGA
    else {

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
#ifdef cFPGA
    else {

    }
#endif

}


void Tensor::set_select(const vector<string>& indices, Tensor *A){
    auto *sd = new SelDescriptor(indices, this->device);
    sd->build(this->shape);
    sd->build_indices();

    // Check if the dimensions of the selection and the tensor are compatibles
    if(sd->oshape==A->shape){
        Tensor::set_select(this, A, sd);
    }else{
        msg("Incompatible dimensions", "Tensor::set_select");
    }
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
#ifdef cFPGA
    else {

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
#ifdef cFPGA
    else {

    }
#endif

}

void Tensor::select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end) {
    ///////////////////////////////////////
    /// Select from A to B, A is bigger
    //////////////////////////////////////

    if ((A->size / A->shape[0]) != (B->size / B->shape[0])) {
        A->info();
        B->info();
        msg("Incompatible shape", "Tensor::select");
    }

    //B->tsem->lock();
    if ((A->isCPU()) && (B->isCPU())) {
        cpu_select(A, B, sind, ini, end);
    }
    else if ((A->isGPU()) && (B->isCPU())) {
        Tensor *Ac=A->clone();
        Ac->toCPU();

        cpu_select(Ac, B, sind, ini, end);

        delete Ac;
    }else if ((A->isCPU()) && (B->isGPU())) {
        Tensor *Bc=B->clone();
        Bc->toCPU();
        cpu_select(A, Bc, sind, ini, end);

        Tensor::copy(Bc,B);

        delete Bc;
    }
    else if ((A->isGPU()) && (B->isGPU())) {
        Tensor *Ac=A->clone();
        Ac->toCPU();

        Tensor *Bc=B->clone();
        Bc->toCPU();

        cpu_select(Ac, Bc, sind, ini, end);

        Tensor::copy(Bc,B);

        delete Ac;
        delete Bc;
    }
    else {
        msg("unsuppoted select", "Tensor::select");
    }
    //B->tsem->unlock();
}
void Tensor::deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end) {
    ///////////////////////////////////////
    /// deSelect from A to B, B is bigger
    //////////////////////////////////////

    if ((A->size / A->shape[0]) != (B->size / B->shape[0])) {
        A->info();
        B->info();
        msg("Incompatible shape", "Tensor::select");
    }

    //B->tsem->lock();
    if ((A->isCPU()) && (B->isCPU())) {
        cpu_deselect(A, B, sind, ini, end);
    }
    else if ((A->isGPU()) && (B->isCPU())) {
        Tensor *Ac=A->clone();
        Ac->toCPU();

        cpu_deselect(Ac, B, sind, ini, end);

        delete Ac;
    }else if ((A->isCPU()) && (B->isGPU())) {
        Tensor *Bc=B->clone();
        Bc->toCPU();
        cpu_deselect(A, Bc, sind, ini, end);

        Tensor::copy(Bc,B);

        delete Bc;
    }
    else if ((A->isGPU()) && (B->isGPU())) {
        Tensor *Ac=A->clone();
        Ac->toCPU();

        Tensor *Bc=B->clone();
        Bc->toCPU();

        cpu_deselect(Ac, Bc, sind, ini, end);
        Tensor::copy(Bc,B);

        delete Ac;
        delete Bc;
    }
    else {
        msg("unsuppoted select", "Tensor::select");
    }
    //B->tsem->unlock();
}

void Tensor::tile(Tensor *A, Tensor *B)
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
