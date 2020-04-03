//Profiling  Kernel


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <ctime>
#include <limits>
#include "tensor.h"

#include "apis/eddlT.h"
#include "layer_core.h"

#include "nn/tensor_nn.h"
#include "../hardware/cpu/nn/cpu_nn.h"
#include "../../src/hardware/fpga/tensor_hls_op.h"
#include "../hardware/cpu/cpu_hw.h"
#include "../../src/hardware/fpga/gemx_wrapper.h"



using namespace std;
using namespace eddlT;

cl::Context context;
cl::CommandQueue q;
cl::CommandQueue com;
cl::Program program;
cl::Kernel kernel_gemx;


int main(int argc, char **argv) {

    typedef short T;
    printf("Profiiling kernel GEMX...\n");
    printf("ARGC =  %d \n",(argc-1));
    for(int i=1; i<argc; i++){
    printf("ARGV[%d]=%s \n", i,argv[i]);   
    }  
    //Read Args
    //A_row, A_col, B_col
    int Ar=atoi(argv[1]);
    int Ac=atoi(argv[2]);
    int Bc=atoi(argv[3]); 
    T A_row=(short)Ar;
    T A_col=(short)Ac;
    T B_col=(short)Bc;
   
    //Create Matrix
    T B_row = A_col;
    T C_row = A_row;
    T C_col = B_col;
    //T A_matrix[A_row][A_col];
    T *A_matrix = new T[A_row * A_col];
    //T B_matrix[B_row][B_col];
    T *B_matrix = new T[B_row * B_col];
    //T C_matrix[C_row][C_col];
    T *C_matrix = new T[C_row * C_col];
    
    printf("A_row = %d \n", A_row);
    printf("A_col = %d \n", A_col);
    printf("B_row = %d \n", B_row);
    printf("B_col = %d \n", B_col);
    printf("C_row = %d \n", C_row);
    printf("C_col = %d \n", C_col);   
     


    //Inicialize Matrix
    //Matrix : A
    printf("Matrix A: \n");
   // for (int i=0; i<A_row; i++){
	//for(int j=0; j<A_col; j++){
        //A_matrix[i * A_col + j] = 1;
	//printf("%d ", A_matrix[i * A_col + j]);
	//}
	//printf("\n");
    //}
    //Matrix : B
    printf("Matrix B: \n");
    //for (int i=0; i<B_row; i++){
	//for(int j=0;j<B_col; j++){
        //B_matrix[i * B_col + j] = 1;
	//printf("%d ", B_matrix[i * B_col + j]);
        //}
	//printf("\n");
    //}
    //Matrix : C
    //printf("Matrix C: \n");
    //for (int i=0; i<C_row; i++){
//	for (int j=0; i<C_col; j++){
        //C_matrix[i * C_col + j] = 1;
	//printf("%d ", C_matrix[i * C_col + j]);
//	}
  //      printf("\n");
    //} 
    
   
   //Initialize context, device and kernel
   cl_int err;
   std::string binaryFile = "/home/jorga20j/Deephealth/eddl/src/hardware/fpga/kernels/xclbin/gemx-short.xclbin";
   unsigned fileBufSize;
   std::vector<cl::Device> devices = xcl::get_xil_devices();
   cl::Device device = devices[0];
   OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
   OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
   char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
   cl::Program::Binaries bins{{fileBuf, fileBufSize}};

   devices.resize(1);
   OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));

   OCL_CHECK(err, kernel_gemx = cl::Kernel(program,"gemxKernel_0", &err));
   //kernel_gemx = clCreateKernel(program(), "gemxKernel_0", &err);
   if (err != CL_SUCCESS) printf("Error creating kernel\n");


   ///Gemx function

   cl_int ret;
   cl::Event event;
   //Gemx setup
   ProfilerStorage gemx_ps("GEMX prof");

   for(int k = 0; k<10; k++){
	   BlockProfiler prof_(gemx_ps);
	   printf("GEMX \n");
	   gemx_setup(A_row,A_col,B_col,A_col,B_col, B_col, B_col);
	   //Create buffers
	   cl::Buffer instr_buffer;
	   cl::Buffer A,B,C;
	   int size_buffer_A = A_row*A_col*sizeof(T);
	   int size_buffer_B = B_row*B_col*sizeof(T);
	   int size_buffer_C = C_row*C_col*sizeof(T);
	   OCL_CHECK(ret,  A = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size_buffer_A, A_matrix, &ret));
	   OCL_CHECK(ret,  B = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size_buffer_B, B_matrix, &ret));
	   OCL_CHECK(ret,  C = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size_buffer_C, C_matrix, &ret));
	   OCL_CHECK(ret, instr_buffer = cl::Buffer(context,CL_MEM_READ_WRITE ,gemx_instr_buffer_size() , nullptr, &ret));
	   //Copy tensors to kernel
	   OCL_CHECK(ret, ret= q.enqueueWriteBuffer(instr_buffer, CL_TRUE, 0,gemx_instr_buffer_size() , gemx_instr_buffer(), nullptr, &event));
	   //// OCL_CHECK(ret, ret = q.enqueueMigrateMemObjects({instr_buffer},CL_MIGRATE_MEM_OBJECT_HOST, NULL, &event));
	   //printf("AFTER MIGRATE %d %d\n", gemx_instr_buffer_size(), gemx_instr_buffer());

	  // OCL_CHECK(ret, q.enqueueMigrateMemObjects(com, 1, &instr_buffer, 0, 0, NULL, &event));

	   OCL_CHECK(ret, ret = q.enqueueCopyBuffer(A, instr_buffer, 0, gemx_page_A() * 4096, size_buffer_A));
	   OCL_CHECK(ret, ret = q.enqueueCopyBuffer(B, instr_buffer, 0, gemx_page_B() * 4096, size_buffer_B));

	   printf("AFTER COPY BUFFER \n");

	   ////OCL_CHECK(ret, ret= q.enqueueWriteBuffer(instr_buffer,CL_TRUE, 0, gemx_instr_buffer_size(), gemx_instr_buffer(), nullptr, nullptr));

	   OCL_CHECK(ret, ret = kernel_gemx.setArg(0, instr_buffer));
	   OCL_CHECK(ret, ret = kernel_gemx.setArg(1, instr_buffer));

	   printf("AFTER SETTING ARGS \n");
	   ///ret = clSetKernelArg(kernel_gemx(), 0, sizeof(cl_mem), (void *)&instr_buffer);
	   ////if (ret != CL_SUCCESS) {printf("Error setting argument\n");}

	   ////ret = clSetKernelArg(kernel_gemx(), 1, sizeof(cl_mem), (void *)&instr_buffer);
	   ////if (ret != CL_SUCCESS) {printf("Error setting argument\n");}
	    
	   OCL_CHECK(ret, ret = q.enqueueTask(kernel_gemx, NULL, &event));
	   //q.finish();
	   printf("AFTER ENQUEUE TASK \n");

	   OCL_CHECK(ret, ret = q.enqueueReadBuffer(instr_buffer, CL_TRUE, gemx_page_A() * 4096, size_buffer_C, C_matrix));
	   OCL_CHECK(ret, ret= q.enqueueCopyBuffer(instr_buffer,C, gemx_page_C() * 4096, 0, size_buffer_C));
	   //printf("AFTER COPYBUFFER \n");
	   q.finish();
   }
    

   //Printing C
   
   //printf("C value = %d \n", C_matrix[0][0]);
   //for(int i =0; i<C_row;i++){
     // for(int j = 0; j<C_col; j++){
      //printf("%d ", C_matrix[i][j]);
      //}
      //printf("\n");
   //}
   gemx_ps.dump();
   printf("The end\n");


   //printf("EL %f %f\n", ptr[0][0], ptr[1][0]);
   //printf("EL %f %f\n", ptr[0][0], ptr[1][0]);
   ////ret = clEnqueueCopyBuffer(q(), instr_buffer,(cl_mem)C->fpga_ptr(), gemx_page_C() * 4096, 0, A->shape[0] * B->shape[1] * sizeof(float), 0, NULL, &event);
   ////if (ret != CL_SUCCESS) {printf("Error setting argument\n");}
   ////ret = clWaitForEvents(1, &event);
   ////if (ret != CL_SUCCESS) {printf("Error setting argument\n");}


   ////clReleaseMemObject(instr_buffer);

}

