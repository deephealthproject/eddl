/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
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
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#endif

#include "eddl/hardware/cpu/cpu_profile.h"

#ifdef cFPGA
extern int next_fpga_tensor_id;
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


Tensor::Tensor(const vector<int> &shape, float *fptr, int dev){
    /*
     * Important! If we are creating a GPU tensor, "fptr" must point to a GPU pointer.
     */
// if NOT define... (I always forget)
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

    fpga_ptr = (cl::Buffer *)nullptr;

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
Tensor::Tensor(const vector<int> &shape, Tensor *T):Tensor(shape,T->ptr, T->device) {}

Tensor::Tensor(const vector<float>& data, const vector<int> &shape, int dev) : Tensor(shape, nullptr, DEV_CPU) {
    // 0. Tensor in CPU

    // 1. Copy data from vector to pointer (CPU)
    std::copy(data.begin(), data.end(), this->ptr);

    // 2. Send to device (if needed)
    if(dev==DEV_CPU) {
        this->updateDevice(dev);
    }else if ((dev >= DEV_GPU) && (dev < DEV_FPGA)) {
        this->toGPU(dev);
    }else{
        msg("Not implemented for FPGA", "Tensor::Tensor");
    }
}

void Tensor::updateDevice(int dev){
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

void Tensor::deleteData(){
    // Careful, you can't know is a pointer is allocated
    if(this->ptr != nullptr){
        if (this->isCPU()) {
            delete this->ptr;
        }
#ifdef cGPU
        else if (this->isGPU())
        {
            gpu_delete_tensor(this->gpu_device, this->ptr);
        }
#endif
#ifdef cFPGA
        else {
      // delete FPGA Tensor
    }
#endif
        this->ptr = nullptr;
    }
}

void Tensor::updateData(float *fptr){
    // TODO: What if the new_pointer is the same?

    if (this->isCPU()) {
        // If null => Reserve memory
        // else => point to data
        if (fptr==nullptr) { this->ptr = get_fmem(this->size,"Tensor::updateData"); _profile_add_tensor(this->size); }
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
        this->gpu_device=this->device-DEV_GPU;
        if (!initcuda[this->gpu_device]){
            gpu_init(this->gpu_device);
            initcuda[this->gpu_device]=1;
        }

<<<<<<< HEAD
        // If null => Reserve memory
        // else => point to data  | CAREFUL! This pointer MUST be a GPU pointer. We cannot check it.
        if (fptr == nullptr) { this->ptr = gpu_create_tensor(gpu_device, this->size); }
        else { this->ptr = fptr; }
=======
          // If null => Reserve memory
          // else => point to data  | CAREFUL! This pointer MUST be a GPU pointer. We cannot check it.
          if (fptr == nullptr) { 
            this->ptr = gpu_create_tensor(gpu_device, this->size);
	  }
          else { this->ptr = fptr; }
>>>>>>> 881b9ab250fb4b403e1694817d24ec46acebcfb1

    }
#endif
#ifdef cFPGA
    else if (isFPGA()) {
        // create FPGA Tensor
        fpga_device = device -DEV_FPGA;
        //gpu_device=device-DEV_GPU;
        if (!initfpga[fpga_device])
          {
           printf("Initializing FPGA device\n");
           fpga_init(/*fpga_device*/);
           printf("Xilinx OpenCL\n");
           initfpga[fpga_device]=1;
         }
        if (fptr == nullptr) {
          #ifdef FPGA_DEBUG
	  printf("  ([updateData fptr==null] creating tensor size %d; id being assigned %d)\n", this->size, next_fpga_tensor_id);
          #endif
	  this->fpga_ptr = fpga_create_tensor(fpga_device, this->size);
	  this->fpga_size = this->size;
	  // we allocate also on cpu so to fluently emulate with cpu
	  this->ptr = get_fmem(this->size,"Tensor::updateData");
	  //
          this->fpga_tensor_id = next_fpga_tensor_id;
          next_fpga_tensor_id++;    
#ifdef FPGA_DEBUG
	  printf("  ([updateData] ptr %p fpga_ptr %p)\n", this->ptr, this->fpga_ptr);
#endif
 	} else { 
	  // The data has already been created in CPU, so we need now to create a buffer in FPGA and write the buffer into it
	  // we first update the cpu buffer
#ifdef FPGA_DEBUG
	  printf("  ([updateData fptr!=null] fptr %p tensor id %d ptr %p fpga_ptr %p size %d fpga_size %d)\n", fptr, this->fpga_tensor_id, this->ptr, this->fpga_ptr, this->size, this->fpga_size);
#endif
	    // is the same pointer, so the updateData simply reassigns the cpu pointer (but we omit it as there is no need)
	    // but if the pointer is different then it means the data changed, in that case we must create a new tensor in fpga and
	    // copy the data
	    //
	    if (this->fpga_ptr == (cl::Buffer *)nullptr) {
	      this->fpga_ptr = fpga_create_tensor(fpga_device, this->size);
	      fpga_size = this->size;
	      fpga_copy_to_fpga(fptr, this);
	      this->fpga_tensor_id = next_fpga_tensor_id++;
#ifdef FPGA_DEBUG
	      printf("    created tensor id %d fpga_ptr %p\n", this->fpga_tensor_id, this->fpga_ptr);
#endif
	    } else {
	      if (this->size != this->fpga_size) {
	        fpga_delete_tensor(fpga_device, this->fpga_ptr, this->fpga_tensor_id, this->fpga_size);
		//
                this->fpga_ptr = fpga_create_tensor(fpga_device, this->size);
		this->fpga_size = this->size;
		fpga_copy_to_fpga(fptr, this);
#ifdef FPGA_DEBUG
		printf("    reallocated tensor id %d new size %d\n", this->fpga_tensor_id, this->fpga_size);
#endif
              } else {
	        //fpga_copy_to_fpga(fptr, this);
#ifdef FPGA_DEBUG
		printf("    just updated the info\n");
#endif
	      }
            }
	    this->ptr = fptr;
	  }
          // For 2 dimensions, map to data to Eigen for efficiency
          // Efficient operations will be done over ptr2, which also points to ptr
          if (this->ndim == 2) {
            this->ptr2=(Eigen::MatrixXf*)new Eigen::Map<Eigen::MatrixXf>(this->ptr, this->shape[1], this->shape[0]);
          }

    }
#endif
}

void Tensor::toCPU(int dev){
#ifdef cGPU
    if (isGPU())
    {

        // Reserve memory for CPU
        float *cpu_ptr = get_fmem(size, "Tensor::toCPU");

        // Copy GPU data to CPU
        gpu_copy_from_gpu(this, cpu_ptr);

        // Delete GPU data
        this->deleteData();

        // Assign CPU pointer
        this->device = dev;  // Must appear after deleting the data
        this->ptr = cpu_ptr;
        if (ndim == 2) {
            ptr2=(Eigen::MatrixXf*)new Eigen::Map<Eigen::MatrixXf>(cpu_ptr, shape[1], shape[0]);
        }

    }
#endif
#ifdef cFPGA
    if (isFPGA()) {
        this->device = dev;

        float *cpu_ptr = get_fmem(size, "Tensor::toCPU");

        if (ndim == 2) {
            ptr2=(Eigen::MatrixXf*)new Eigen::Map<Eigen::MatrixXf>(cpu_ptr, shape[1], shape[0]);
        }

        fpga_copy_from_fpga(this, cpu_ptr);
        this->ptr = cpu_ptr;
        fpga_delete_tensor(fpga_device,this->fpga_ptr, this->fpga_tensor_id, this->size);
    }
#endif
}

void Tensor::toGPU(int dev){
#ifdef cGPU
    if (isCPU()) {
        this->device = dev;
        this->gpu_device = this->device - DEV_GPU;

        float *cpu_ptr = ptr;
        float *gpu_ptr = gpu_create_tensor(this->gpu_device, this->size);

        if (!initcuda[gpu_device]){
            gpu_init(gpu_device);
            initcuda[gpu_device] = 1;
        }

        this->ptr = gpu_ptr;
        gpu_copy_to_gpu(cpu_ptr, this);
        delete cpu_ptr;
    }
    else if (isGPU())
    {
//        printf("Tensor already in GPU\n");
    }
#endif
#ifdef cFPGA
    if (isFPGA()) {
      printf("Error, toGPU when using FPGA\n"); exit(1);
    }
#endif
}

Tensor* Tensor::clone(){
    auto* t_new = new Tensor(this->shape, this->device);
    Tensor::copy(this, t_new);
    return t_new;
}

void Tensor::reallocate(Tensor* old_t, vector<int> *s){
    // Update values
    if(s != nullptr){
        updateDevice(old_t->device);
        updateShape(*s);
        updateSize();
        updateStrides();
    }

    updateData(old_t->ptr);
}

Tensor::~Tensor() {
<<<<<<< HEAD
    this->deleteData();
=======
    if (isCPU()) {
        delete ptr;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_delete_tensor(gpu_device, ptr);
      }
#endif
#ifdef cFPGA
    else {
      // delete FPGA Tensor
      fpga_delete_tensor(fpga_device, fpga_ptr, fpga_tensor_id, size);
      delete ptr;
    }
#endif
>>>>>>> 881b9ab250fb4b403e1694817d24ec46acebcfb1
    delete tsem;
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

string Tensor::getDeviceName(){
    if ((this->device >= DEV_CPU) && (this->device < DEV_GPU)) { return "CPU"; }
    else if ((device >= DEV_GPU) && (this->device < DEV_FPGA)) { return "GPU"; }
    else if (this->device >= DEV_FPGA) { return "FPGA"; }
    return "unknown";
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
void Tensor::resize(int b, float *fptr) {
    if (b == shape[0]) return;

    // Get new shape
    vector<int> new_shape = this->getShape();
    new_shape[0] = b;

    // Update attributes
    updateShape(new_shape);
    updateSize();
    updateStrides();
    if (fptr == nullptr) deleteData();  // Potential error
    updateData(fptr);
}

