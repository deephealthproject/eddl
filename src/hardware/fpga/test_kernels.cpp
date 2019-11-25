/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include "xcl2.hpp"
#include <vector>
#include <math.h>
#define DATA_SIZE 5000 
// Kernels with a tensor as input
#define SET_LOG 0x0
#define SET_EXP 0x1
#define SET_SQRT 0x2
#define SET_SQR 0x3
#define TOTAL_SUM 0x4
#define TOTAL_ABS 0x5
// Kernels with a float as input
#define SUM 0x10
#define MULT 0x20
#define RAND_UNIFORM 0x30
#define RAND_SUNIFORM 0x40
#define RAND_BINARY 0x50
// Kernels with 2 floats as inputs
#define RAND_GAUSSIAN 0x100

static inline uint64_t get_cycles()
{
  uint64_t t;
  __asm volatile ("rdtsc" : "=A"(t));
  return t;
}


int main()
{
    
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::CommandQueue com;
    cl::Program program;
    cl::Kernel tensor_op;
    cl::Kernel multitensor_op;
    cl::Kernel kernel_sum6;
    cl::Kernel mult2D;
    cl::Kernel sum2D_rowwise;
    cl::Kernel kernel_cent;
    cl::Kernel relu_soft_d;
    cl::Kernel reduce_sum2D;
    cl::Kernel kernel_accuracy;
    cl::Kernel kernel_total_sum;

//   std::string binaryFile = "/home/carherlu/DEEPHEALTH/EDDLL/src/fpga/kernels/xclbin/tensor_op.hw.xilinx_u200_xdma_201830_1.xclbin";
    std::string binaryFile = "kernels/xclbin/tensor_op.sw_emu.xilinx_u200_xdma_201830_1.xclbin";
    unsigned fileBufSize;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};

    devices.resize(1);
    OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));
    OCL_CHECK(err, tensor_op= cl::Kernel(program,"tensor_op", &err));
    OCL_CHECK(err, multitensor_op = cl::Kernel(program,"multitensor_op", &err));
    /*OCL_CHECK(err, kernel_sum6 = cl::Kernel(program,"kernel_sum6", &err));
    OCL_CHECK(err, mult2D = cl::Kernel(program,"kernel_mult2D", &err));
    OCL_CHECK(err, sum2D_rowwise = cl::Kernel(program,"kernel_sum2D_rowwise", &err));
    OCL_CHECK(err, kernel_cent = cl::Kernel(program,"kernel_cent", &err));
    OCL_CHECK(err, relu_soft_d = cl::Kernel(program,"relu_soft_d", &err));
    OCL_CHECK(err, reduce_sum2D = cl::Kernel(program,"reduce_sum2D", &err));
    OCL_CHECK(err, kernel_accuracy = cl::Kernel(program,"kernel_accuracy", &err));
    OCL_CHECK(err, kernel_total_sum = cl::Kernel(program,"kernel_total_sum", &err));
*/

    cl::Event event;
    int tam = 80;
    float *tensor_in = (float*) malloc (tam*sizeof(float));

    OCL_CHECK(err, cl::Buffer buffer_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
            tam*sizeof(float), tensor_in, &err));
    OCL_CHECK(err, cl::Buffer buffer_in2   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
            tam*sizeof(float), tensor_in, &err));;


    OCL_CHECK(err, err = tensor_op.setArg(0, buffer_in));
    OCL_CHECK(err, err = tensor_op.setArg(1, (float)1.333));
    OCL_CHECK(err, err = tensor_op.setArg(2, tam));
    OCL_CHECK(err, err = tensor_op.setArg(3, 8)); //KERNEL-ID
    

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in},0/* 0 means from host*/));
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in2},0/* 0 means from host*/));
    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is recommended
    // to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(err, err = q.enqueueTask(tensor_op, NULL, &event));
    q.finish();
    // Copy Result from Device Global Memory to Host Local Memory
    printf("HOLA\n");
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
// OPENCL HOST CODE AREA END
    OCL_CHECK(err, err = multitensor_op.setArg(0, buffer_in));
    OCL_CHECK(err, err = multitensor_op.setArg(1, buffer_in2));
    OCL_CHECK(err, err = multitensor_op.setArg(2, tam));
    OCL_CHECK(err, err = multitensor_op.setArg(3, tam));
    OCL_CHECK(err, err = multitensor_op.setArg(4, 9));

    OCL_CHECK(err, err = q.enqueueTask(multitensor_op, NULL, &event));
    q.finish();

    // Compare the results of the Device to the simulation
    // CALL the tensor class here and compare with the original operations

}
