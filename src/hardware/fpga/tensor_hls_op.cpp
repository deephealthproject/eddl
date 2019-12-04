/**********
**********/

#include "libs/xcl2.hpp"
#include <vector>
#include <math.h>
#include "tensor_hls_op.h"
#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

//#include <thrust/device_ptr.h>
//#include <thrust/transform.h>
//#include <thrust/reduce.h>
//#include <thrust/functional.h>
//#include <thrust/extrema.h>

//#define DBG_FPGA

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
cl::Kernel kernel_normalize;
cl::Kernel el_div;

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

void fpga_init(){ // initialize only once
   
    cl_int err;
    std::string binaryFile = "/home/carherlu/DEEPHEALTH/LASTVERSION/eddl/src/hardware/fpga/kernels/xclbin/tensor_op.hw.xilinx_u200_xdma_201830_2.xclbin";
    unsigned fileBufSize;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    
    devices.resize(1);
    OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));
    OCL_CHECK(err, tensor_op= cl::Kernel(program,"tensor_op", &err));
    OCL_CHECK(err, multitensor_op = cl::Kernel(program,"multitensor_op", &err));
    //OCL_CHECK(err, kernel_sum6 = cl::Kernel(program,"kernel_sum6", &err));
    OCL_CHECK(err, mult2D = cl::Kernel(program,"kernel_mult2D", &err));
    OCL_CHECK(err, sum2D_rowwise = cl::Kernel(program,"kernel_sum2D_rowwise", &err));
    //OCL_CHECK(err, kernel_cent = cl::Kernel(program,"kernel_cent", &err)); 
    //OCL_CHECK(err, relu_soft_d = cl::Kernel(program,"relu_soft_d", &err));
    //OCL_CHECK(err, reduce_sum2D = cl::Kernel(program,"reduce_sum2D", &err));
    OCL_CHECK(err, kernel_accuracy = cl::Kernel(program,"kernel_accuracy", &err));
    OCL_CHECK(err, kernel_total_sum = cl::Kernel(program,"kernel_total_sum", &err));
    OCL_CHECK(err, el_div = cl::Kernel(program,"el_div", &err));
    OCL_CHECK(err, kernel_normalize = cl::Kernel(program,"kernel_normalize", &err));

}

void fpga_create_tensor(Tensor *T, int dev)
{
    cl_int err;
    int size = T->size;
    //cl::Buffer buf; 
    //printf("Creating Buffer at ref %d -- size %d\n", 0, size);

    OCL_CHECK(err,T->fpga_ptr = cl::Buffer(context,CL_MEM_READ_WRITE, size*sizeof(float), NULL, &err));

    //OCL_CHECK(err, err= q.enqueueWriteBuffer(T->fpga_ptr, CL_TRUE, 0, T->tam*sizeof(float), ptr, nullptr, nullptr));
    //verify2(T->fpga_ptr, T->tam);


    //T->fpga_ptr = &buf;
    //printf("Creating Buffer at ref %d -- %d size %d\n", buf,(T->fpga_ptr), size);
}

void fpga_delete_tensor(Tensor *T)
{

//  T->fpga_ptr.release();

}



void close_fpga(){
 //delete fileBuf;
}

///////////////////////////////////////////
void fpga_copy_fpga(Tensor *A, Tensor *B)
{
    cl_int err;
    OCL_CHECK(err, err= q.enqueueCopyBuffer((A->fpga_ptr), (B->fpga_ptr), 0, 0, A->size*sizeof(float)));
    q.finish();
}

void fpga_copy_to_fpga(float *nptr, Tensor *A)
{
    cl_int err;
    cl::Event blocking_event;
    OCL_CHECK(err, err= q.enqueueWriteBuffer((A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(float), nptr, nullptr, &blocking_event));
    q.finish();
    //blocking_event.wait();
    //printf("Copy Tensor with tam %d in Buffer ref %d -- %f\n", A->tam, A->fpga_ptr,*nptr);
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({A->fpga_ptr},0/* 0 means from host*/));
    
}

///////////////////////////////////////////
void fpga_copy_from_fpga(Tensor *A,float *nptr)
{
    cl_int err;
    cl::Event event;
    OCL_CHECK(err, err= q.enqueueReadBuffer((A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(float), nptr, nullptr, &event));
    q.finish();;
//    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST));
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

void fpga_tensor_sum6(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC)
{
    cl_int err;
    cl::Event event;
    #ifdef DBG_FPGA 
        printf("FPGA::SUM6\n");
    #endif

    OCL_CHECK(err, err = kernel_sum6.setArg(0, scA));
    OCL_CHECK(err, err = kernel_sum6.setArg(1, (A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sum6.setArg(2, scB));
    OCL_CHECK(err, err = kernel_sum6.setArg(3, (B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sum6.setArg(4, (C->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sum6.setArg(5, incC));
    OCL_CHECK(err, err = kernel_sum6.setArg(6, A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_sum6, NULL, &event));
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



void fpga_reduce_sum2D( Tensor *A, Tensor *B, int axis,int incB)
{
    cl_int err;
    cl::Event event;

    #ifdef DBG_FPGA 
        printf("FPGA::REDUCESUM2D\n");
    #endif

    OCL_CHECK(err, err = reduce_sum2D.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = reduce_sum2D.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = reduce_sum2D.setArg(2, A->shape[0]));
    OCL_CHECK(err, err = reduce_sum2D.setArg(3, A->shape[1]));
    OCL_CHECK(err, err = reduce_sum2D.setArg(4, axis));
    OCL_CHECK(err, err = reduce_sum2D.setArg(5, incB));

    OCL_CHECK(err, err = q.enqueueTask(reduce_sum2D, NULL, &event));
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


void fpga_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C){
    
    cl_int err;
    cl::Event event;

    #ifdef DBG_FPGA 
        printf("FPGA::SUM2DROWWISE\n");
    #endif
    
    OCL_CHECK(err, err = sum2D_rowwise.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = sum2D_rowwise.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = sum2D_rowwise.setArg(2, (C->fpga_ptr)));
    OCL_CHECK(err, err = sum2D_rowwise.setArg(3, A->shape[0]));
    OCL_CHECK(err, err = sum2D_rowwise.setArg(4, A->shape[1])); 

    OCL_CHECK(err, err = q.enqueueTask(sum2D_rowwise, NULL, &event));
    q.finish();
}

void fpga_mult2D(Tensor *A,int tA, Tensor *B, int tB, Tensor *C, int incC){
   
    cl_int err; 
    cl::Event event;
    /*if (incC != 0) {
       printf("WARNING:: Mult2D with inc not supported\n"); 
    }else printf("REGULAR MATMULT\n");*/

    #ifdef DBG_FPGA 
        printf("FPGA::MULT2D\n");
    #endif

    OCL_CHECK(err, err = mult2D.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = mult2D.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = mult2D.setArg(2, (C->fpga_ptr)));
    OCL_CHECK(err, err = mult2D.setArg(3, A->shape[0]));
    OCL_CHECK(err, err = mult2D.setArg(4, A->shape[1]));
    OCL_CHECK(err, err = mult2D.setArg(5, B->shape[0]));
    OCL_CHECK(err, err = mult2D.setArg(6, B->shape[1]));
    OCL_CHECK(err, err = mult2D.setArg(7, tA));
    OCL_CHECK(err, err = mult2D.setArg(8, tB));
    
    //printf("sizes A(%dx%d) B(%dx%d) C(%dx%d)\n", A->sizes[0], A->sizes[1], B->sizes[0], B->sizes[1],C->sizes[0], C->sizes[1]);
    //OCL_CHECK(err, err = tensor_op.setArg(5, incC));  
    OCL_CHECK(err, err = q.enqueueTask(mult2D, NULL, &event));
 //   event.wait();
    q.finish();
}

float fpga_total_sum (Tensor *A){
   cl_int err;
   cl::Event event, result_ready;

   #ifdef DBG_FPGA 
        printf("FPGA::TOTAL_SUM\n");
    #endif

   float *sum = (float*) malloc(sizeof(float));
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


int fpga_accuracy (Tensor *A, Tensor *B){
   cl_int err;
   cl::Event event, result_ready;


   #ifdef DBG_FPGA 
        printf("FPGA::ACCURACY\n");
    #endif
 
   int *acc = (int*) malloc(sizeof(int));
   *acc = 0;
   OCL_CHECK(err, cl::Buffer a(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 4 ,acc, &err));
 
   OCL_CHECK(err, err = kernel_accuracy.setArg(0, (A->fpga_ptr)));
   OCL_CHECK(err, err = kernel_accuracy.setArg(1, (B->fpga_ptr)));
   OCL_CHECK(err, err = kernel_accuracy.setArg(2, A->shape[0]));
   OCL_CHECK(err, err = kernel_accuracy.setArg(3, A->shape[1]));
   OCL_CHECK(err, err = kernel_accuracy.setArg(4, a));

   OCL_CHECK(err, err = q.enqueueTask(kernel_accuracy, NULL, &event));
   event.wait();
 
   OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a},CL_MIGRATE_MEM_OBJECT_HOST, NULL, &result_ready));
   result_ready.wait();
   return *acc;
   
}

void fpga_cent(Tensor *A, Tensor *B, Tensor *C){
    cl_int err; 
    cl::Event event;
    
    #ifdef DBG_FPGA 
        printf("FPGA::CENT\n");
    #endif

    OCL_CHECK(err, err = kernel_cent.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_cent.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_cent.setArg(2, (C->fpga_ptr)));
    OCL_CHECK(err, err = kernel_cent.setArg(3, A->size));
    
    OCL_CHECK(err, err = q.enqueueTask(kernel_cent, NULL, &event));
  //  event.wait();  
    q.finish();

}

void fpga_relu_soft_d(Tensor *D, Tensor *I, Tensor *PD, int kernel_id){
    
    cl_int err;
    cl::Event event;

    #ifdef DBG_FPGA 
        printf("FPGA::RELU_SOFT\n", kernel_id);
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
        printf("FPGA::TENSOR_OPERATION %d\n", kernel_id);
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

void fpga_reduction(ReduceDescriptor *RD){
    float val,sum;
    int ind;
    int d;
    int i,j,k,l,s;

    // [MEAN]: Compute items to be reduced
    if (RD->m==0) {
       d=1;
       for(i=0;i<RD->axis.size();i++){
           d *= RD->I->shape[RD->axis[i]];
       }
    }
    //reduce
   for(i=0;i<RD->index.size();i++)
   {
       sum=0;
        for(j=0;j<RD->index[i].size();j++) {
            float v=RD->I->ptr[RD->index[i][j]];
           if (RD->m==2) {
               if (j==0) {val=v;ind=RD->index[i][j];}
               else if (v>val) {
                   val=v;
                   ind=RD->index[i][j];
               }
           }
           else if (RD->m==3) {
             if (j==0) {val=v;ind=RD->index[i][j];}
             else if (v<val) {
                 val=v;
                 ind=RD->index[i][j];
             }
           }
           else sum+=v;
       }
       // set in Output
       if (RD->m<2) { // mean or sum
           if (RD->m==0) sum/=d;
           if (RD->keepdims) {
               for(j=0;j<RD->index[i].size();j++) {
                   RD->O->ptr[RD->index[i][j]]=sum;
                 }
           }
           else RD->O->ptr[i]=sum;
       }
       else { // max or min
           if (RD->keepdims) {
               for(j=0;j<RD->index[i].size();j++) {
                   RD->O->ptr[RD->index[i][j]]=val;
                   RD->S->ptr[RD->index[i][j]]=ind;
               }
           }
           else {
               RD->O->ptr[i]=val;
               RD->S->ptr[i]=ind;
           }
       }
    }// i
}
