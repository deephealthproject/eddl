/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <iostream>
#include <iomanip>
#include <stdexcept>

#include "eddl/tensor/tensor.h"
#include "eddl/utils.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_nn.h"
#endif

using namespace std;

// TODO: Don't like here
int initcuda[MAX_GPUS] = {0, 0, 0, 0, 0, 0, 0, 0};
int linpos;
extern ostream &operator<<(ostream &os, const vector<int> shape);


Tensor::Tensor() : device(DEV_CPU), ndim(0), size(0) {}

Tensor::Tensor(const vector<int> &shape, float *fptr, int dev){
    /*
     * Important! If we are creating a GPU tensor, "fptr" must point to a GPU pointer.
     */
#ifndef cGPU
    if ((dev > DEV_CPU)&&(dev<DEV_FPGA)) {
        throw std::runtime_error("Not compiled for GPU");
    }
#endif
#ifndef cFPGA
    if (dev >= DEV_FPGA) {
        throw std::runtime_error("Not compiled for FPGA");
    }
#endif

    // Update values
    updateDevice(dev);
    updateShape(shape);
    updateSize();
    updateStrides();
    updateData(fptr);

    this->tsem = new mutex();
}

// From shape and device
Tensor::Tensor(const vector<int> &shape, int dev):Tensor(shape, nullptr, dev){}

// From shape and Tensor (sharing ptr)
Tensor::Tensor(const vector<int> &shape, Tensor *T):Tensor(shape,T->ptr,T->device) {}


void Tensor::updateDevice(unsigned int dev){
    this->device = dev;
}

void Tensor::updateShape(const vector<int> &new_shape){
    this->shape = vector<int>(new_shape);
    this->ndim = this->shape.size();
}

void Tensor::updateSize() {
    this->size = 1;

    for(auto &d : this->shape) {
        this->size *= d;
    }
}

void Tensor::updateStrides() {
    this->stride.clear();  // Remove all elements

    unsigned long int new_size = this->size;
    for(int i=0;i<ndim;i++) {
        new_size /= shape[i];
        this->stride.push_back(new_size);
    }
}

void Tensor::deleteData() {
    if (this->ptr != nullptr) {
        if (this->isCPU()) {
            delete this->ptr;
        }
#ifdef cGPU
        else if (this->isGPU()) {
            free_gpu_ptr(this->ptr, this->gpu_device);
        }
#endif
#ifdef cFPGA
        else {
            // delete FPGA Tensor
        }
#endif
    }

    this->ptr = nullptr;
}

void Tensor::updateData(float *fptr){

    if (this->isCPU()) {
        // If null => Reserve memory
        // else => point to data
        if (fptr==nullptr) { this->ptr = get_fmem(this->size,"Tensor::updateData"); }
        else { this->ptr = fptr; };

        // For 2 dimensions, map to data to Eigen for efficiency
        // Efficient operations will be done over ptr2, which also points to ptr
        if (this->ndim == 2) {
            this->ptr2=(Eigen::MatrixXf*)new Eigen::Map<Eigen::MatrixXf>(this->ptr, this->shape[1], this->shape[0]);
        }
    }
#ifdef cGPU
    else if (this->isGPU())
    {
        gpu_device=this->device-DEV_GPU;
        if (!initcuda[gpu_device]){
            gpu_init(gpu_device);
            initcuda[gpu_device]=1;
        }

        // If null => Reserve memory
        // else => point to data  | CAREFUL! This pointer MUST be a GPU pointer. We cannot check it.
        if (fptr == nullptr) { this->ptr = gpu_create_tensor(gpu_device, this->size); }
        else { this->ptr = fptr; }

    }
#endif
#ifdef cFPGA
    else {
        // create FPGA Tensor
      }
#endif
}

void Tensor::toCPU(int dev){
#ifdef cGPU
    if (isGPU())
    {
        // Reserve CPU memory and get GPU pointer
        float *cpu_ptr = get_fmem(this->size, "Tensor::toCPU");
        float *gpu_ptr = this->ptr;

        // Copy GPU data to a CPU pointer
        gpu2cpu(gpu_ptr, cpu_ptr, this->size, this->gpu_device);

        // Delete current data (ptr)
        this->deleteData();

        // Update pointer
        this->updateDevice(dev);
        this->updateData(cpu_ptr);
    }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::toGPU(int dev){
#ifdef cGPU
    if (this->isCPU()) {
        // Set device
        this->gpu_device = (int)this->device - DEV_GPU;

        // Initialize device
        if (!initcuda[this->gpu_device]){
            gpu_init(this->gpu_device);
            initcuda[this->gpu_device] = 1;
        }

        // Get CPU pointer and reserve GPU memory
        float *cpu_ptr = this->ptr;
        float *gpu_ptr = get_gpu_fmem(this->size, this->gpu_device);

        // Copy CPU data to a GPU pointer
        cpu2gpu(cpu_ptr, gpu_ptr, this->size, this->gpu_device);

        // Delete current data (ptr)
        this->deleteData();

        // Update pointer
        this->updateDevice(dev);
        this->updateData(gpu_ptr);

    }
    else if (this->isGPU())
      {
//        printf("Tensor already in GPU\n");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::clone(){
    auto* t_new = new Tensor(this->shape, this->device);
    Tensor::copy(this, t_new);
    return t_new;
}

void Tensor::reallocate(Tensor* old_t, vector<int>* new_shape){
    // Update values
    if(new_shape != nullptr){
        this->updateDevice(old_t->device);
        this->updateShape(*new_shape);
        this->updateSize();
        this->updateStrides();
    }

    // Delete previous data and point to the new data
    this->deleteData();
    this->updateData(old_t->ptr);
}

Tensor::~Tensor() {
    this->deleteData();
    delete tsem;
}

int Tensor::isCPU() const { return (device == DEV_CPU); }

int Tensor::isGPU() const { return ((device >= DEV_GPU) && (device < DEV_FPGA)); }

int Tensor::isFPGA() const { return (device >= DEV_FPGA); }

vector<int> Tensor::getShape() {
    return vector<int>(this->shape);
}

void Tensor::info() {
    int cols = 15;
    cout << "-------------------------------" << endl;
    cout << setw(cols) << left << "class: "        << "Tensor" << endl;
    cout << setw(cols) << left << "ndim: "         << this->ndim << endl;
    cout << setw(cols) << left << "shape: "        << "(" << printVector<int>(this->shape) << ")" << endl;
    cout << setw(cols) << left << "strides: "      << "(" << printVector<int>(this->stride) <<  ")" << endl;
    cout << setw(cols) << left << "itemsize: "     << this->size << endl;
    cout << setw(cols) << left << "contiguous: "   << true << endl; // for future
    cout << setw(cols) << left << "order: "        << 'C' << endl;  // C=>C order, F=>Fortran order
    cout << setw(cols) << left << "data pointer: " << &this->ptr << endl;
    cout << setw(cols) << left << "type: "         << "float" << " (" << sizeof(float) << " bytes)" << endl;
    cout << setw(cols) << left << "device: "         << this->getStrDevice() << " (code = " << this->device << ")" << endl;
    cout << "-------------------------------" << endl;
}


/**
  *  @brief Prints the content of the tensor
  *
  *  @param precision Number of decimals places to use
  *  @param raw  Print the tensor without format
*/
void Tensor::print(int precision, bool raw) {
    int opened = 0;
    int closed = 0;

    // Clone to CPU (if needed)
    Tensor *aux = nullptr;
    if (this->isCPU()) {
        aux = this;
    }else{
        aux = new Tensor(this->shape, DEV_CPU);
        Tensor::copy(this, aux);
    }

    // ***** Shitty code to prettify the output *******
    std::stringstream buffer;
    buffer << std::fixed;
    buffer << std::setprecision(precision);

    for (int i = 0; i < aux->size; ++i) {

        if(raw){
            // Print number
            buffer << aux->ptr[i] << " ";

        }else{

            // Open brackets
            opened = 0;
            for (int j = 0; j < aux->ndim-1; ++j) {
                if(i%aux->stride[j]==0){
                    if(!opened && closed==1){ if(ndim==2){ buffer << "\n"; } else { buffer << " "; } }
                    buffer << "[";
                    opened += 1;
                }
            }

            // Print number
            buffer << aux->ptr[i];

            // Close brackets
            closed = 0;
            for (int j = 0; j < aux->ndim-1; ++j) {
                if((i+1)%aux->stride[j]==0) {
                    buffer << "]";
                    closed += 1;
                }
            }

            // Break lines
            if (i+1 < aux->size){
                if(!closed){ buffer << " ";}
                else{
                    if (closed == 2 ) {  buffer << "\n"; }
                    else if (closed == 3) { buffer << "\n\n"; }
                    else if (closed > 3) { buffer << "\n\n\n"; }
                }
            }

        }

    }

    // Print to buffer
    if(aux->ndim>1){
        cout << "[\n" << buffer.str() << "\n]" << endl;  // For readability
    }else{
        cout << "[" << buffer.str() << "]" << endl;
    }

    // Free memory
    if (!this->isCPU()) {
        delete aux;
    }
}

string Tensor::getStrDevice(){
    if ((this->device >= DEV_CPU) && (this->device < DEV_GPU)) { return "CPU"; }
    else if ((device >= DEV_GPU) && (this->device < DEV_FPGA)) { return "GPU"; }
    else if (this->device >= DEV_FPGA) { return "FPGA"; }
    return "unknown";
}

int Tensor::get_mode(string mode){
    if(mode == "constant"){
        // (k k k k | a b c d | k k k k)
        // The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
        return 0;
    }else if(mode == "reflect"){
        // (d c b a | a b c d | d c b a)
        // The input is extended by reflecting about the edge of the last pixel.
        return 1;
    }else if(mode == "nearest"){
        // (a a a a | a b c d | d d d d)
        // The input is extended by replicating the last pixel.
        return 2;
    }else if(mode == "mirror"){
        // (d c b | a b c d | c b a)
        // The input is extended by reflecting about the center of the last pixel.
        return 3;
    }else if(mode == "wrap"){
        // (a b c d | a b c d | a b c d)
        // The input is extended by wrapping around to the opposite edge.
        return 4;
    }else if(mode == "original"){
        // (o o o o | a b c d | o o o o)
        // The input is extended by filling all values beyond the edge with the original values
        return 5;
    }else {  // constant
        return -1;
    }
}


bool Tensor::isSquared(Tensor *A){
    int last_dim = A->shape[0];
    for(int i=0; i<A->ndim; i++){
        if(last_dim!=A->shape[i]){
            return false;
        }
    }
    return true;
}
