/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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
#endif

using namespace std;

// TODO: Don't like here
int initcuda[MAX_GPUS] = {0, 0, 0, 0, 0, 0, 0, 0};
int initfpga[MAX_FPGAS] = {0, 0, 0, 0, 0, 0, 0, 0};
int linpos;
extern ostream &operator<<(ostream &os, const vector<int> shape);

void checkCompatibility(Tensor *A, Tensor *B, const string &title){
    if (A->device != B->device) {
        msg("Tensors in different devices", title);
    }

    if (!Tensor::sameShape(A, B)){
        msg("Tensors with different shape", title);
    }
}


void checkCompatibility(Tensor *A, Tensor *B, Tensor *C, const string &title){
    checkCompatibility(A, B, title);
    checkCompatibility(A, C, title);
}




Tensor::Tensor() : device(DEV_CPU), ndim(0), size(0) {}


Tensor::Tensor(const vector<int> &shape, float *fptr, int dev, void *fptr2){
    /*
     * Important! If we are creating a GPU tensor, "fptr" must point to a GPU pointer.
     */
// if NOT define... (I always forget)
#ifndef cGPU
    if ((dev > DEV_CPU)&&(dev<DEV_FPGA)) {
        throw std::runtime_error("Not compiled for GPU");
    }
#endif

    fpga_ptr = nullptr;   // valgrind complaints here

    // Update values
    updateDevice(dev);
    updateShape(shape);
    updateSize();
    updateStrides();
    updateData(fptr, fptr2);
}

// From shape and device
Tensor::Tensor(const vector<int> &shape, int dev):Tensor(shape, nullptr, dev){}

// From shape and Tensor (sharing ptr)
Tensor::Tensor(const vector<int> &shape, Tensor *T) : Tensor(shape,T->ptr, T->device) {}

Tensor::Tensor(const vector<float>& data, const vector<int> &shape, int dev) : Tensor(shape, nullptr, DEV_CPU) {
    isshared=false;
    // 0. Tensor in CPU

    // 1. Copy data from vector to pointer (CPU)
    std::copy(data.begin(), data.end(), this->ptr);

    // 2. Send to device (if needed)
    if(dev==DEV_CPU) {
        this->updateDevice(dev);
    }else if ((dev >= DEV_GPU) && (dev < DEV_FPGA)) {
        this->toGPU(dev);
    }else{
	    this->toFPGA(dev);
    }
}

Tensor::~Tensor() {
    this->deleteData();
}

void Tensor::updateDevice(int dev){
    this->device = dev;
}

void Tensor::updateShape(const vector<int> &new_shape){
    // this->shape = vector<int>(new_shape);
    this->shape.clear();
    for (int _ : new_shape) this->shape.push_back(_);
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

void Tensor::deleteData(){
    // Carefpdal, you can't know is a pointer is allocated
//fprintf(stderr, "control passed %s(%d) %p %p %p \n", __FILE__, __LINE__, this, this->ptr, this->ptr2);
    if (isshared) {
        if (/*this->isCPU() && this->ndim == 2 &&*/ this->ptr2 != nullptr) {
            delete this->ptr2;
            this->ptr2 = nullptr;
        }
        return;
    }

    if(this->ptr != nullptr){
        if (this->isCPU()) {
            // Delete eigen matrix
            if (/*this->ndim == 2 &&*/ this->ptr2 != nullptr){
                delete this->ptr2; //double free or corruption (out)
                this->ptr2 = nullptr;
            }
            eddl_free(this->ptr); // because currently memory for tensor data is allocated by means of posix_memalign()
            this->ptr = nullptr;
        }
#ifdef cGPU
        else if (this->isGPU())
        {
            gpu_delete_tensor(this->gpu_device, this->ptr);
            //cout<<"delete here"<<endl;
        }
#endif
        // Set pointer to null
        this->ptr = nullptr;
    }
}


void Tensor::updateData(float *fptr, void *fptr2, bool setshared){
    // TODO: What if the new_pointer is the same?
    // Solved with setshared for reshape_

    bool was_shared = isshared;
    isshared=false;
    if (this->isCPU()) {
        // If null => Reserve memory
        // else => point to data
        if (fptr==nullptr) {
            if (false == was_shared && this->ptr != nullptr) eddl_free(this->ptr);
            this->ptr = get_fmem(this->size,"Tensor::updateData");
        } else {
            this->ptr = fptr; isshared=setshared;
        };

        if (this->ptr2 != nullptr) {
            delete this->ptr2;
            this->ptr2 = nullptr;
        }
        // For 2 dimensions, map to data to Eigen for efficiency
        // Efficient operations will be done over ptr2, which also points to ptr
        if (this->ndim == 2){
            this->ptr2 = new Eigen::Map<Eigen::MatrixXf>(this->ptr, this->shape[1], this->shape[0]);

        }
    }
#ifdef cGPU
    else if (this->isGPU())
    {
        this->gpu_device=this->device-DEV_GPU;
        if (!initcuda[this->gpu_device]){
            gpu_init(this->gpu_device);
            initcuda[this->gpu_device]=1;
        }

        // If null => Reserve memory
        // else => point to data  | CAREFUL! This pointer MUST be a GPU pointer. We cannot check it.
        if (fptr == nullptr) { this->ptr = gpu_create_tensor(this->gpu_device, this->size); }
        else { this->ptr = fptr; isshared=setshared;}
    }
#endif
}

void Tensor::toCPU(int dev){
    if (this->isCPU()) {
        // printf("Tensor already in CPU\n");
    }else{
        float *cpu_ptr = nullptr;

#ifdef cGPU
        if (this->isGPU()){
            // Reserve memory for CPU
            cpu_ptr = get_fmem(size, "Tensor::toCPU");

            // Copy GPU data to CPU
            gpu_copy_from_gpu(this, cpu_ptr);
        }
#endif
        // COMMON: Delete data in HW + reassign pointer
        // Delete data
        this->deleteData();

        // Assign CPU pointer

        this->device = dev;  // Must appear after deleting the data

        this->updateData(cpu_ptr,nullptr,false);
    }


}

void Tensor::toGPU(int dev){
    // TODO: Improve this with existing functions
#ifdef cGPU
    if (this->isCPU()) {
        this->device = dev;
        this->gpu_device = this->device - DEV_GPU;

        float *cpu_ptr = this->ptr;
        float *gpu_ptr = gpu_create_tensor(this->gpu_device, this->size);

        if (!initcuda[gpu_device]){
            gpu_init(gpu_device);
            initcuda[gpu_device] = 1;
        }

        this->ptr = gpu_ptr;
        gpu_copy_to_gpu(cpu_ptr, this);
        eddl_free(cpu_ptr); // because currently memory for tensor data is allocated by means of posix_memalign()
        if (/*this->ndim == 2 &&*/ this->ptr2 != nullptr){
            delete this->ptr2;
            this->ptr2 = nullptr;
        }
    }
    else if (this->isGPU())
    {
//        printf("Tensor already in GPU\n");
    }
#endif
}

void Tensor::toFPGA(int dev){
    printf("toFPGA Deprecated\n"); exit(1);
#ifdef cGPU
    printf("Error, toFPGA with  cGPU implementation not supported\n");
    exit(1);
#endif
}

void Tensor::toDevice(int dev){
    int dev_id = Tensor::getDeviceID(dev);

    // Select device
    if(dev_id == 0){  // CPU
        this->toCPU(dev);
    }else if(dev_id == 1){  // GPU
        this->toGPU(dev);
    }else if(dev_id == 2) {  // FPGA
        this->toFPGA(dev);
    }else{
        throw std::runtime_error("Not compiled for FPGA");
    }
}

Tensor* Tensor::clone(){
    auto* t_new = new Tensor(this->shape, this->device);
    Tensor::copy(this, t_new);
    return t_new;
}

void Tensor::reallocate(Tensor* old_t){
    Tensor::reallocate(old_t, {});
}


void Tensor::reallocate(Tensor* old_t, const vector<int> &shape){
    // Update values
    if(!shape.empty()){
        updateDevice(old_t->device);
        updateShape(shape);
        updateSize();
        updateStrides();
    }

    // Not recommended
    updateData(old_t->ptr,nullptr,false);
}

int Tensor::isCPU() { return (device == DEV_CPU); }

int Tensor::isGPU() { return ((device >= DEV_GPU) && (device < DEV_FPGA)); }

int Tensor::isFPGA() { return (device >= DEV_FPGA); }

vector<int> Tensor::getShape() {
    return vector<int>(this->shape);
}

unsigned int Tensor::numel(){
    return (unsigned int)this->size;
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
    cout << setw(cols) << left << "is shared: " << isshared << endl;
    cout << setw(cols) << left << "type: "         << "float" << " (" << sizeof(float) << " bytes)" << endl;
    cout << setw(cols) << left << "device: " << this->getDeviceName() << " (code = " << this->device << ")" << endl;
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

    int lines = 0;
    int max_lines = 100000;
    for (int i = 0; i < aux->size; ++i) {
        if(i % this->stride[0]==0){lines++;}

        if(raw){
            // Print number
            buffer << aux->ptr[i] << ", ";

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

            // Stop
            if(lines >= max_lines){
                cout << "Maximum tensor length exceeded." << endl;
                cout << "Printing only first " << max_lines << " rows:" << endl;
                break;
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

string Tensor::getDeviceName() const{
    int dev_id = Tensor::getDeviceID(this->device);
    if (dev_id == 0) { return "CPU"; }
    else if (dev_id == 1) { return "GPU"; }
    else if (dev_id == 2) { return "FPGA"; }
    return "unknown";
}

int Tensor::getDeviceID(int dev){
    if ((dev >= DEV_CPU) && (dev < DEV_GPU)) { return 0; }
    else if ((dev >= DEV_GPU) && (dev < DEV_FPGA)) { return 1; }
    else if (dev >= DEV_FPGA) { return 2; }
    return -1;
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

// Resizing tensors
void Tensor::resize(int b, float *fptr, void *fptr2, bool delete_data) {
    if (b == shape[0]) return;

    // Get new shape
    vector<int> new_shape = this->getShape();
    new_shape[0] = b;

    // Update attributes
    updateShape(new_shape);
    updateSize();
    updateStrides();
    if (!isshared && delete_data) deleteData();  // Potential error on layers such as Reshape (passed pointer)
    updateData(fptr, fptr2);
}

vector<string> Tensor::hardware_supported() {
    // Get supported hardware
    vector<string> hw_supported = {"cpu"};
#ifdef cGPU
    hw_supported.emplace_back("cuda");
#ifdef cCUDNN
    hw_supported.emplace_back("cudnn");
#endif
#endif
#ifdef cFPGA
    hw_supported.emplace_back("fpga");
#endif

    return hw_supported;
}

bool Tensor::is_hardware_supported(string hardware){
    vector<string> hw_supported = Tensor::hardware_supported();
    if(hardware=="gpu") { hardware = "cuda"; }  // gpu could be both "cuda" and "cudnn"
    bool hw_found = std::find(hw_supported.begin(), hw_supported.end(), hardware) != hw_supported.end();
    return hw_found;
}
