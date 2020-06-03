/**********
**********/

#include "eddl/hardware/fpga/xcl2.hpp"
#include <vector>
#include <math.h>
#include "eddl/hardware/fpga/tensor_hls_op.h"
#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"
//#include "gemx_wrapper.h"

//#include <thrust/device_ptr.h>
//#include <thrust/transform.h>
//#include <thrust/reduce.h>
//#include <thrust/functional.h>
//#include <thrust/extrema.h>

#define DBG_FPGA

extern cl::Context context;
extern cl::CommandQueue q;
extern cl::CommandQueue com;
extern cl::Program program;
extern cl::Kernel tensor_op;
extern cl::Kernel multitensor_op;
extern cl::Kernel kernel_add;


extern cl::Kernel relu_soft_d;


extern cl::Kernel kernel_total_sum;
extern cl::Kernel kernel_normalize;
extern cl::Kernel el_div;
//extern cl::Kernel kernel_gemx;
extern cl::Kernel kernel_core;

//cl::Buffer buffer;

uint64_t get_duration_ns (const cl::Event &event) {
    cl_int err;
    uint64_t nstimestart, nstimeend;
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend));
    return(nstimeend-nstimestart);
}

void verify2(cl::Buffer &buffer, int size) {

    cl_int err;
    float *ptr = (float *) malloc (size*sizeof(float));
    OCL_CHECK(err, err = q.enqueueReadBuffer(buffer, CL_TRUE, 0, size*sizeof(int), ptr));
    int correct = 1;
    for (int i=0; i<size; i++){
        if (ptr[i] < 1.299) {correct = 0; printf("%f ", ptr[i]);break;}
    }
    if (correct) printf("Verified buffer\n"); else printf("Something WENT wrong \n");
}

void verify(Tensor *T) {
    cl_int err;
    printf("veryfying");
    float *ptr = (float *) malloc (T->size*sizeof(float));
    OCL_CHECK(err, err = q.enqueueReadBuffer(T->fpga_ptr, CL_TRUE, 0, T->size*sizeof(int), ptr));
    for (int i=0; i<T->size; i++){
	printf("ptr = %f \n", ptr[i]);
    }
}







void tensor_op_hls(Tensor *A, float fp, int kernel_id)
{
    cl_int err;
    cl::Event task_end;

    #ifdef DBG_FPGA
        printf("TENSOR_OP_HLS %d\n", kernel_id);
    #endif

//    printf("Tensor with XX size %d in Buffer ref %d -- %d\n", A->size, 0,kernel_id);

    OCL_CHECK(err, tensor_op= cl::Kernel(program,"tensor_op", &err));
    OCL_CHECK(err, err = tensor_op.setArg(0, A->fpga_ptr));
    OCL_CHECK(err, err = tensor_op.setArg(1, fp));
    OCL_CHECK(err, err = tensor_op.setArg(2, A->size));
    OCL_CHECK(err, err = tensor_op.setArg(3, kernel_id));


    OCL_CHECK(err, err = q.enqueueTask(tensor_op, NULL, &task_end));
    //printf("Tensor with XX size %d in Buffer ref %d -- %d\n", A->size, 0/*A->fpga_ptr*/,kernel_id);
    //verify(A);
    q.finish();
}

void fpga_tensor_add(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC)
{
    cl_int err;
    cl::Event event;
    #ifdef DBG_FPGA
        printf("FPGA::add\n");
    #endif

    OCL_CHECK(err, err = kernel_add.setArg(0, scA));
    OCL_CHECK(err, err = kernel_add.setArg(1, (A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_add.setArg(2, scB));
    OCL_CHECK(err, err = kernel_add.setArg(3, (B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_add.setArg(4, (C->fpga_ptr)));
    OCL_CHECK(err, err = kernel_add.setArg(5, incC));
    OCL_CHECK(err, err = kernel_add.setArg(6, A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_add, NULL, &event));
    event.wait();
}

void fpga_tensor_normalize(Tensor *A, float min, float max)
{
    cl_int err;
    cl::Event event;

    #ifdef DBG_FPGA
        printf("FPGA::NORMALIZE\n");
    #endif
    float min_ori = 0;//gpu_min(A);
    float max_ori = 1;//gpu_max(A);


    OCL_CHECK(err, err = kernel_normalize.setArg(0, A->fpga_ptr));
    OCL_CHECK(err, err = kernel_normalize.setArg(1, max));
    OCL_CHECK(err, err = kernel_normalize.setArg(2, min));
    OCL_CHECK(err, err = kernel_normalize.setArg(3, max_ori));
    OCL_CHECK(err, err = kernel_normalize.setArg(4, min_ori));
    OCL_CHECK(err, err = kernel_normalize.setArg(5, A->size));
    OCL_CHECK(err, err = q.enqueueTask(kernel_normalize, NULL, &event));
    q.finish();
}





void fpga_el_div_mult(Tensor *A, Tensor *B, Tensor *C, int incC, int op) {
   cl_int err;
   cl::Event event;

   #ifdef DBG_FPGA
        printf("FPGA::EL_VIV_MULT %d\n", op);
   #endif

   OCL_CHECK(err, err = el_div.setArg(0, (A->fpga_ptr)));
   OCL_CHECK(err, err = el_div.setArg(1, (B->fpga_ptr)));
   OCL_CHECK(err, err = el_div.setArg(2, (C->fpga_ptr)));
   OCL_CHECK(err, err = el_div.setArg(3, A->size));
   OCL_CHECK(err, err = el_div.setArg(4, incC));
   OCL_CHECK(err, err = el_div.setArg(5, op));

   OCL_CHECK(err, err = q.enqueueTask(el_div, NULL, &event));
   q.finish();



//  for (int i = 0; i < A->size; i++)
//    if (incC) C->ptr[i] += A->ptr[i] / B->ptr[i];
//    else C->ptr[i] = A->ptr[i] / B->ptr[i];
}




float fpga_total_sum (Tensor *A){
   cl_int err;
   cl::Event event, result_ready;

   #ifdef DBG_FPGA
        printf("FPGA::TOTAL_SUM\n");
    #endif

   float *sum = (float*) malloc(sizeof(float));
   *sum = 0;
   OCL_CHECK(err, cl::Buffer a(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(float) ,sum, &err));

   OCL_CHECK(err, err = kernel_total_sum.setArg(0, (A->fpga_ptr)));
   OCL_CHECK(err, err = kernel_total_sum.setArg(1, A->size));
   OCL_CHECK(err, err = kernel_total_sum.setArg(2, a));
   OCL_CHECK(err, err = q.enqueueTask(kernel_total_sum, NULL, &event));
   event.wait();
   OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a},CL_MIGRATE_MEM_OBJECT_HOST, NULL, &result_ready));
   result_ready.wait();
   return *sum;


}



void fpga_relu_soft_d(Tensor *D, Tensor *I, Tensor *PD, int kernel_id){

    cl_int err;
    cl::Event event;

    #ifdef DBG_FPGA
        //printf("FPGA::RELU_SOFT\n", kernel_id);
    #endif


    OCL_CHECK(err, err = relu_soft_d.setArg(0, (D->fpga_ptr)));
    OCL_CHECK(err, err = relu_soft_d.setArg(1, (I->fpga_ptr)));
    OCL_CHECK(err, err = relu_soft_d.setArg(2, (PD->fpga_ptr)));
    OCL_CHECK(err, err = relu_soft_d.setArg(3, D->size));
    OCL_CHECK(err, err = relu_soft_d.setArg(4, kernel_id));

    OCL_CHECK(err, err = q.enqueueTask(relu_soft_d, NULL, &event));
    q.finish();
}


void fpga_tensor_operation(Tensor *A, Tensor *B, int kernel_id)
{
    cl_int err;
    cl::Event event;
    int aux;
    if (kernel_id == FPGARELU) 	{aux = A->size;}
    else {
	     if (kernel_id == FPGASOFTM) {aux = A->shape[0];}
        else{printf("Tensor Operation not supported %d\n", kernel_id); exit(1);}
    }

    #ifdef DBG_FPGA
        //printf("FPGA::TENSOR_OPERATION %d\n", kernel_id);
    #endif

    OCL_CHECK(err, err = multitensor_op.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = multitensor_op.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = multitensor_op.setArg(2, aux));
    OCL_CHECK(err, err = multitensor_op.setArg(3, A->shape[1]));
    OCL_CHECK(err, err = multitensor_op.setArg(4, kernel_id));

    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is recommended
    // to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(err, err = q.enqueueTask(multitensor_op, NULL, &event));
    q.finish();
}


void fpga_core(Tensor *A, float v, int kernel_id)
{
    cl_int err;
    cl::Event event;

    #ifdef DBG_FPGA
        printf("FPGA::fpga_core %d\n", kernel_id);
    #endif

    OCL_CHECK(err, err = kernel_core.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_core.setArg(1, (A->size)));
    OCL_CHECK(err, err = kernel_core.setArg(2, v));
    OCL_CHECK(err, err = kernel_core.setArg(3, kernel_id));

    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is recommended
    // to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(err, err = q.enqueueTask(kernel_core, NULL, &event));
    q.finish();
}
