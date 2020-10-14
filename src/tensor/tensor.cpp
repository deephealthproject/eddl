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
#ifndef cFPGA
    if (dev >= DEV_FPGA) {
        throw std::runtime_error("Not compiled for FPGA");
    }
#endif

#ifdef cFPGA
    fpga_ptr = (cl::Buffer *)nullptr;
#endif

    // Update values
    updateDevice(dev);
    updateShape(shape);
    updateSize();
    updateStrides();
    updateData(fptr, fptr2);

    this->tsem = new mutex();
}

// From shape and device
Tensor::Tensor(const vector<int> &shape, int dev):Tensor(shape, nullptr, dev){}

// From shape and Tensor (sharing ptr)
Tensor::Tensor(const vector<int> &shape, Tensor *T) : Tensor(shape,T->ptr, T->device
#ifdef cFPGA
		, (void *)T->fpga_ptr
#endif
		) {}

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
    if(this->tsem != nullptr) { this->tsem->unlock(); delete tsem; }
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
    // Carefpdal, you can't know is a pointer is allocated
    if (isshared) return;

    if(this->ptr != nullptr){
        if (this->isCPU()) {
            // Delete eigen matrix
            if (this->ndim == 2){
                delete this->ptr2;
                delete[] this->ptr;
                this->ptr2 = nullptr;
                this->ptr = nullptr;  // Redundant
            }else{
                delete[] this->ptr;
                this->ptr = nullptr;  // Redundant
            }

        }
#ifdef cGPU
        else if (this->isGPU())
        {
            gpu_delete_tensor(this->gpu_device, this->ptr);
        }
#endif
#ifdef cFPGA
        else if (this->isFPGA())
	{
            fpga_delete_tensor(this->fpga_device, this->fpga_ptr, this->fpga_tensor_id, this->fpga_size);
	}

      // delete FPGA Tensor
#endif
        // Set pointer to null
        this->ptr = nullptr;
    }
}

void Tensor::updateData(float *fptr, void *fptr2,bool setshared){
    // TODO: What if the new_pointer is the same?
    // Solved with setshared for reshape_
    isshared=false;
    if (this->isCPU()) {
        // If null => Reserve memory
        // else => point to data
        if (fptr==nullptr) { this->ptr = get_fmem(this->size,"Tensor::updateData"); }
        else { this->ptr = fptr; isshared=setshared;};

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
#ifdef cFPGA
    else if (this->isFPGA())
    {
	#ifdef FPGA_DEBUG
	printf("Tensor::updateData: fptr=%p, fptr2=%p, setshared=%d\n", fptr, fptr2, setshared);
        #endif
        fpga_device = device-DEV_FPGA;
        if (!initfpga[fpga_device]) {
          #ifdef FPGA_DEBUG
          printf(" initializing FPGA device\n");
          #endif
          fpga_init(/*fpga_device*/);
          initfpga[fpga_device]=1;
        }
        if (fptr == nullptr) {
          #ifdef FPGA_DEBUG
          printf(" creating tensor: size=%d fpga_tensor_id=%d\n", this->size, next_fpga_tensor_id);
          #endif
          this->fpga_ptr = fpga_create_tensor(fpga_device, this->size);
          this->fpga_size = this->size;
          // we allocate also on cpu so to fluently emulate with cpu
          this->ptr = get_fmem(this->size,"Tensor::updateData");
          //
          this->fpga_tensor_id = next_fpga_tensor_id;
          next_fpga_tensor_id++;
          #ifdef FPGA_DEBUG
          printf("  new pointers: ptr=%p fpga_ptr=%p\n", this->ptr, this->fpga_ptr);
          #endif
        } else {
          printf(" info: fpga_ptr %p fptr2 %p\n", this->fpga_ptr, fptr2);
          if ((this->fpga_ptr == (cl::Buffer *)nullptr) && (fptr2 == nullptr)) {
            this->fpga_ptr = fpga_create_tensor(fpga_device, this->size);
            this->fpga_size = this->size;
            this->fpga_tensor_id = next_fpga_tensor_id;
            next_fpga_tensor_id++;
            fpga_copy_to_fpga(fptr, this);
            #ifdef FPGA_DEBUG
            printf("  fpga_ptr and fptr2 were null, we create a buffer with tensor id %d\n", this->fpga_tensor_id);
            #endif
          } else if ((this->fpga_ptr == (cl::Buffer *)nullptr) && (fptr2 != nullptr)) {
            #ifdef FPGA_DEBUG
            printf("  fpga_ptr null but fptr2 not\n");
            #endif
            this->fpga_size = this->size;
            this->fpga_ptr = (cl::Buffer *)fptr2;
	    this->fpga_tensor_id = next_fpga_tensor_id;
	    next_fpga_tensor_id++;
            #ifdef FPGA_DEBUG
	    printf("   new fpga_size %d fpga_ptr %p fpga_tensor_id %d\n", this->fpga_size, this->fpga_ptr, this->fpga_tensor_id);
            #endif
          } else {
            #ifdef FPGA_DEBUG
	    printf("  fpga_ptr and fptr2 are not null\n");
            #endif
            this->fpga_size = this->size;
            this->fpga_ptr = (cl::Buffer *)fptr2;
            #ifdef FPGA_DEBUG
	    printf("   new fpga_size %d fpga_ptr %x\n", this->fpga_size, this->fpga_ptr);
            #endif
          }
          #ifdef FPGA_DEBUG
          printf("  end of changes: fptr %p tensor id %d ptr %p fpga_ptr %p size %d fpga_size %d fptr2 %p)\n", fptr, this->fpga_tensor_id, this->ptr, this->fpga_ptr, this->size, this->fpga_size, fptr2);
          #endif
          this->ptr = fptr;
        }
        // For 2 dimensions, map to data to Eigen for efficiency
        // Efficient operations will be done over ptr2, which also points to ptr
        if (this->ndim == 2) {
          this->ptr2= new Eigen::Map<Eigen::MatrixXf>(this->ptr, this->shape[1], this->shape[0]);
        }
        #ifdef FPGA_DEBUG
        printf("-------------------------\n");
        #endif
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
#ifdef cFPGA
        if (isFPGA())
    {

        // Reserve memory for CPU
        float *cpu_ptr = get_fmem(size, "Tensor::toCPU");

        // Copy GPU data to CPU
        fpga_copy_from_fpga(this, cpu_ptr);
    }
#endif
        // COMMON: Delete data in HW + reassign pointer
        // Delete data
        this->deleteData();

        // Assign CPU pointer
        this->device = dev;  // Must appear after deleting the data
        this->updateData(cpu_ptr);
    }


}

void Tensor::toGPU(int dev){
    // TODO: Improve this with existing functions
#ifdef cGPU
    if (this->isCPU()) {
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
    else if (this->isGPU())
    {
//        printf("Tensor already in GPU\n");
    }
#endif
#ifdef cFPGA
    printf("Error, toGPU with  cFPGA implementation not supported\n");
    exit(1);
#endif
}

void Tensor::toFPGA(int dev){
#ifdef cFPGA
    if (this->isCPU()) {
        this->device = dev;
        this->fpga_device = this->device - DEV_FPGA;

        float *cpu_ptr = ptr;
	cl::Buffer *fpga_ptr = fpga_create_tensor(this->fpga_device, this->size);

        if (!initfpga[fpga_device]){
            fpga_init(/*fpga_device*/);
            initfpga[fpga_device] = 1;
        }

        this->fpga_ptr = fpga_ptr;
        fpga_copy_to_fpga(cpu_ptr, this);
	// we do not remove the cpu_ptr as is used for cpuemu mode
        //delete cpu_ptr;
    }
    else if (this->isFPGA())
    {
//        printf("Tensor already in FPGA\n");
    }
#endif
#ifdef cGPU
    printf("Error, toFPGA with  cGPU implementation not supported\n");
    exit(1);
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

    // Not recommended
    updateData(old_t->ptr);
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
void Tensor::resize(int b, float *fptr, void *fptr2, bool delete_data) {
    if (b == shape[0]) return;

    // Get new shape
    vector<int> new_shape = this->getShape();
    new_shape[0] = b;

    // Update attributes
    updateShape(new_shape);
    updateSize();
    updateStrides();
    if (fptr != nullptr && delete_data) deleteData();  // Potential error on layers such as Reshape (passed pointer)
    updateData(fptr, fptr2);
}
