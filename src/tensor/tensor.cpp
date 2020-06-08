/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
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

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#endif

#ifdef cFPGA
extern int next_fpga_tensor_id;
#endif

using namespace std;

// TODO: Don't like here
int initcuda[MAX_GPUS] = {0, 0, 0, 0, 0, 0, 0, 0};
int initfpga[MAX_FPGAS] = {0, 0, 0, 0, 0, 0, 0, 0};
int linpos;
extern ostream &operator<<(ostream &os, const vector<int> shape);

/**
  *  @brief Constructor of an uninitialized tensor in CPU with dimension and size 0
  *  @return a tensor
*/
Tensor::Tensor() : device(DEV_CPU), ndim(0), size(0) {}

/**
  *  @brief Constructor of an uninitialized tensor
  *
  *  @param shape Vector of ints specifying the shape of the tensor
  *  @param fptr  memory pointer
  *  @param dev  One of DEV_CPU or DEV_GPU
  *  @return a tensor
*/
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

    tsem = new mutex();
}


// From shape and device
/**
  *  @brief Constructor of an uninitialized tensor
  *
  *  @param shape Vector of ints specifying the shape of the tensor
  *  @param dev  One of DEV_CPU or DEV_GPU
  *  @return a tensor
*/
Tensor::Tensor(const vector<int> &shape, int dev):Tensor(shape, nullptr, dev){}

// From shape and Tensor (sharing ptr)
/**
  *  @brief Constructor of an uninitialized tensor
  *
  *  @param shape Vector of ints specifying the shape of the tensor
  *  @param T  tensor from which to take device and memory pointer
  *  @return a tensor
*/
Tensor::Tensor(const vector<int> &shape, Tensor *T):Tensor(shape,T->ptr,T->device) {}


/**
  *  @brief Moves the tensor to new computing device
  *
  *  @param dev  One of DEV_CPU or DEV_GPU
*/
void Tensor::updateDevice(int dev){
    this->device = dev;
}

/**
  *  @brief Change the shape of a tensor
  *
  *  @param shape  Vector of ints specifying the new shape of the tensor
*/
void Tensor::updateShape(const vector<int> &shape){
    this->shape = vector<int>(shape);
    this->ndim = this->shape.size();
}


/**
  *  @brief Update the size of the tensor
*/
void Tensor::updateSize() {
    this->size = 1;

    for(auto &d : this->shape) {
        this->size *= d;
    }
}


/**
  *  @brief Update the strides of the tensor
*/
void Tensor::updateStrides() {
    this->stride.clear();  // Remove all elements

    int new_size = this->size;
    for(int i=0;i<ndim;i++) {
        new_size /= shape[i];
        this->stride.push_back(new_size);
    }
}

/**
  *  @brief Delete tensor data
*/
void Tensor::deleteData(){
    if(this->ptr != nullptr){
        delete this->ptr;
        this->ptr = nullptr;
    }
}

/**
  *  @brief Update tensor data
  *  
  *  @param fptr Pointer to the new data
*/
void Tensor::updateData(float *fptr){

    if (isCPU()) {
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
    else if (isGPU())
        {
          gpu_device=this->device-DEV_GPU;
          if (!initcuda[gpu_device]){
              gpu_init(gpu_device);
              initcuda[gpu_device]=1;
          }

          // If null => Reserve memory
          // else => point to data  | CAREFUL! This pointer MUST be a GPU pointer. We cannot check it.
          if (fptr == nullptr) { 
            this->ptr = gpu_create_tensor(gpu_device, this->size);
	  }
          else { this->ptr = fptr; }

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
        //printf("Creating FPGA tensor\n");
        if (fptr == nullptr) {
          #ifdef FPGA_DEBUG
	  printf("FPGA: creating tensor with id %d\n", next_fpga_tensor_id);
          #endif
	  this->fpga_ptr = fpga_create_tensor(fpga_device, this->size);
	  // we allocate also on cpu so to fluently emulate with cpu
	  this->ptr = get_fmem(this->size,"Tensor::updateData");
	  //
          this->fpga_tensor_id = next_fpga_tensor_id;
          next_fpga_tensor_id++;    
 	}
	else { 
	  // The data has already been created in CPU, so we need now to create a buffer in FPGA and write the buffer into it
	  this->fpga_ptr = fpga_create_tensor(fpga_device, this->size);
	  fpga_copy_to_fpga(fptr, this);
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

/**
  *  @brief Move the tensor to CPU
  *  
  *  @param dev target CPU device
*/
void Tensor::toCPU(int dev){
#ifdef cGPU
    if (isGPU())
      {
        this->device = dev;

        float *cpu_ptr = get_fmem(size, "Tensor::toCPU");
        float *gpu_ptr = ptr;

        if (ndim == 2) {
            ptr2=(Eigen::MatrixXf*)new Eigen::Map<Eigen::MatrixXf>(cpu_ptr, shape[1], shape[0]);
        }

        gpu_copy_from_gpu(this, cpu_ptr);
        this->ptr = cpu_ptr;
        gpu_delete_tensor(gpu_device,gpu_ptr);

      }
#endif
#ifdef cFPGA
    if (isFPGA()) {
      printf("toCPU not implemented yet\n"); exit(1);

    }
#endif
}

/**
  *  @brief Move the tensor to GPU
  *  
  *  @param dev target GPU device
*/
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


/**
  *  @brief Obtain a copy of a tensor
  *  
  *  @return The copied tensor
*/
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
    }
#endif
    delete tsem;
}

/**
  *  @brief Check if a tensor is in CPU
  *  
  *  @return 1 if the tensor is in CPU, 0 otherwise
*/
int Tensor::isCPU() { return (device == DEV_CPU); }

/**
  *  @brief Check if a tensor is in GPU
  *  
  *  @return 1 if the tensor is in GPU, 0 otherwise
*/
int Tensor::isGPU() { return ((device >= DEV_GPU) && (device < DEV_FPGA)); }


/**
  *  @brief Check if a tensor is in FPGA
  *  
  *  @return 1 if the tensor is in FPGA, 0 otherwise
*/
int Tensor::isFPGA() { return (device >= DEV_FPGA); }

/**
  *  @brief Get the shape of a tensor
  *  
  *  @return a vector of ints representing the shape
*/
vector<int> Tensor::getShape() {
    return vector<int>(this->shape);
}

/**
  *  @brief Print information about a tensor
*/
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

/**
  *  @brief Check the device where a tensor is
  *  
  *  @return String. One of "CPU", "GPU", "FPGA" or "unknown"
*/
string Tensor::getStrDevice(){
    if ((this->device >= DEV_CPU) && (this->device < DEV_GPU)) { return "CPU"; }
    else if ((device >= DEV_GPU) && (this->device < DEV_FPGA)) { return "GPU"; }
    else if (this->device >= DEV_FPGA) { return "FPGA"; }
    return "unknown";
}

/**
  *  @brief Obtain a flag indicating the tensor's mode. A tensor can be in one of the following modes:
  *  "constant": The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter. E.g.: (k k k k | a b c d | k k k k)
  *  "reflect": The input is extended by reflecting about the edge of the last pixel. E.g.:(d c b a | a b c d | d c b a)
  *  "nearest": The input is extended by replicating the last pixel. E.g.: (a a a a | a b c d | d d d d)
  *  "mirror": The input is extended by reflecting about the center of the last pixel. E. g.: (d c b | a b c d | c b a)
  *  "wrap": The input is extended by wrapping around to the opposite edge. E.g.: (a b c d | a b c d | a b c d)
  *  "original": The input is extended by filling all values beyond the edge with the original values. E.g.: (o o o o | a b c d | o o o o)
  *  @return integer. 0 if constant, 1 if reflect, 2 if nearest, 3 id mirror, 4 if wrap, 5 if original. Otherwise -1.
*/
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

/**
  *  @brief Check if a tensor is squared or not
  *  @return boolean. True if the tensor is squared, False otherwise.
*/
bool Tensor::isSquared(Tensor *A){
    int last_dim = A->shape[0];
    for(int i=0; i<A->ndim; i++){
        if(last_dim!=A->shape[i]){
            return false;
        }
    }
    return true;
}
