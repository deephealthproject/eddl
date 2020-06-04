/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#include "eddl/hardware/fpga/xcl2.hpp"
#include <vector>
#include <math.h>
#include "eddl/hardware/fpga/tensor_hls_op.h"
#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"
#include "eddl/hardware/fpga/fpga_hw.h"
#include <sys/time.h>
#include "eddl/hardware/cpu/cpu_hw.h"

cl::Context      context;
cl::CommandQueue q;
cl::CommandQueue com;
cl::Program      program;
cl::Kernel       tensor_op;
cl::Kernel       multitensor_op;
cl::Kernel       kernel_add;
cl::Kernel       mult2D;
cl::Kernel       sum2D_rowwise;
cl::Kernel       kernel_cent;
cl::Kernel       relu_soft_d;
cl::Kernel       reduce_sum2D;
cl::Kernel       kernel_accuracy;
cl::Kernel       kernel_total_sum;
cl::Kernel       kernel_normalize;
cl::Kernel       el_div;
//cl::Kernel       kernel_gemx;
cl::Kernel       kernel_core;

// profiling
int num_instances_fpga[_NUM_FPGA_FUNCS];
float mb_memory_needed_fpga;

// profiling functions
void _profile_fpga_funcname(int i, char *name) {
  switch(i) {
      case _FPGA_ALL              : strcpy(name, "all"); break;
      case _FPGA_ANY              : strcpy(name, "any"); break;
      case _FPGA_ISFINITE         : strcpy(name, "isfinite"); break;
      case _FPGA_ISINF            : strcpy(name, "isinf"); break;
      case _FPGA_ISNAN            : strcpy(name, "isnan"); break;
      case _FPGA_ISNEGINF         : strcpy(name, "isneginf"); break;
      case _FPGA_ISPOSINF         : strcpy(name, "isposinf"); break;
      case _FPGA_LOGICAL_AND      : strcpy(name, "logical_and"); break;
      case _FPGA_LOGICAL_OR       : strcpy(name, "logical_or"); break;
      case _FPGA_LOGICAL_NOT      : strcpy(name, "logical_not"); break;
      case _FPGA_LOGICAL_XOR      : strcpy(name, "logical_xor"); break;
      case _FPGA_ALLCLOSE         : strcpy(name, "allclose"); break;
      case _FPGA_ISCLOSE          : strcpy(name, "isclose"); break;
      case _FPGA_GREATER          : strcpy(name, "greater"); break;
      case _FPGA_GREATER_EQUAL    : strcpy(name, "greater_equal"); break;
      case _FPGA_LESS             : strcpy(name, "less"); break;
      case _FPGA_LESS_EQUAL       : strcpy(name, "less_equal"); break;
      case _FPGA_EQUAL            : strcpy(name, "equal"); break;
      case _FPGA_NOT_EQUAL        : strcpy(name, "not_equal"); break;
      case _FPGA_EQUAL2           : strcpy(name, "equal2"); break;
      case _FPGA_TRANSPOSE        : strcpy(name, "transpose"); break;
      case _FPGA_COPY             : strcpy(name, "copy"); break;
      case _FPGA_FILL_            : strcpy(name, "fill_"); break;
      case _FPGA_FILL             : strcpy(name, "fill"); break;
      case _FPGA_SELECT           : strcpy(name, "select"); break;
      case _FPGA_SELECT_BACK      : strcpy(name, "select_back"); break;
      case _FPGA_SET_SELECT       : strcpy(name, "set_select"); break;
      case _FPGA_SET_SELECT_BACK  : strcpy(name, "set_select_back"); break;
      case _FPGA_SELECT2          : strcpy(name, "select2"); break;
      case _FPGA_DESELECT         : strcpy(name, "deselect"); break;
      case _FPGA_CONCAT           : strcpy(name, "concat"); break;
      case _FPGA_RANGE            : strcpy(name, "range"); break;
      case _FPGA_EYE              : strcpy(name, "eye"); break;
      case _FPGA_SINGLE_SHIFT     : strcpy(name, "single_shift"); break;
      case _FPGA_SINGLE_ROTATE    : strcpy(name, "single_rotate"); break;
      case _FPGA_SINGLE_SCALE     : strcpy(name, "single_scale"); break;
      case _FPGA_SINGLE_FLIP      : strcpy(name, "single_flip"); break;
      case _FPGA_SINGLE_CROP      : strcpy(name, "single_crop"); break;
      case _FPGA_SINGLE_CROP_SCALE : strcpy(name, "single_crop_scale"); break;
      case _FPGA_SHIFT            : strcpy(name, "shift"); break;
      case _FPGA_ROTATE           : strcpy(name, "rotate"); break;
      case _FPGA_SCALE            : strcpy(name, "scale"); break;
      case _FPGA_CROP             : strcpy(name, "crop"); break;
      case _FPGA_CROP_SCALE       : strcpy(name, "crop_scale"); break;
      case _FPGA_SHIFT_RANDOM     : strcpy(name, "shift_random"); break;
      case _FPGA_ROTATE_RANDOM    : strcpy(name, "rotate_random"); break;
      case _FPGA_SCALE_RANDOM     : strcpy(name, "scale_random"); break;
      case _FPGA_FLIP_RANDOM      : strcpy(name, "flip_random"); break;
      case _FPGA_CROP_RANDOM      : strcpy(name, "crop_random"); break;
      case _FPGA_CROP_SCALE_RANDOM     : strcpy(name, "crop_scale_random"); break;
      case _FPGA_CUTOUT_RANDOM    : strcpy(name, "cutout_random"); break;
      case _FPGA_RAND_UNIFORM     : strcpy(name, "rand_uniform"); break;
      case _FPGA_RAND_SIGNED_UNIFORM : strcpy(name, "rand_signed_uniform"); break;
      case _FPGA_BINARY           : strcpy(name, "binary"); break;
      case _FPGA_RAND_NORMAL      : strcpy(name, "rand_normal"); break;
      case _FPGA_ABS_             : strcpy(name, "abs_"); break;
      case _FPGA_ACOS_            : strcpy(name, "acos_"); break;
      case _FPGA_ADD_             : strcpy(name, "add_"); break;
      case _FPGA_ASIN_            : strcpy(name, "asin_"); break;
      case _FPGA_ATAN_            : strcpy(name, "atan_"); break;
      case _FPGA_CEIL_            : strcpy(name, "ceil_"); break;
      case _FPGA_CLAMP_           : strcpy(name, "clamp_"); break;
      case _FPGA_COS_             : strcpy(name, "cos_"); break;
      case _FPGA_COSH_            : strcpy(name, "cosh_"); break;
      case _FPGA_EXP_             : strcpy(name, "exp_"); break;
      case _FPGA_FLOOR_           : strcpy(name, "floor_"); break;
      case _FPGA_INV_             : strcpy(name, "inv_"); break;
      case _FPGA_LOG_             : strcpy(name, "log_"); break;
      case _FPGA_LOG2_            : strcpy(name, "log2_"); break;
      case _FPGA_LOG10_           : strcpy(name, "log10_"); break;
      case _FPGA_LOGN_            : strcpy(name, "logn_"); break;
      case _FPGA_MOD_             : strcpy(name, "mod_"); break;
      case _FPGA_MULT_            : strcpy(name, "mult_"); break;
      case _FPGA_NORMALIZE_       : strcpy(name, "normalize_"); break;
      case _FPGA_POW_             : strcpy(name, "pow_"); break;
      case _FPGA_POWB_            : strcpy(name, "powb_"); break;
      case _FPGA_RECIPROCAL_      : strcpy(name, "reciprocal_"); break;
      case _FPGA_REMAINDER_       : strcpy(name, "remainder_"); break;
      case _FPGA_ROUND_           : strcpy(name, "round_"); break;
      case _FPGA_RSQRT_           : strcpy(name, "rsqrt_"); break;
      case _FPGA_SIGMOID_         : strcpy(name, "sigmoid_"); break;
      case _FPGA_SIGN_            : strcpy(name, "sign_"); break;
      case _FPGA_SIN_             : strcpy(name, "sin_"); break;
      case _FPGA_SINH_            : strcpy(name, "sinh_"); break;
      case _FPGA_SQR_             : strcpy(name, "sqr_"); break;
      case _FPGA_SQRT_            : strcpy(name, "sqrt_"); break;
      case _FPGA_TAN_             : strcpy(name, "tan_"); break;
      case _FPGA_TANH_            : strcpy(name, "tanh_"); break;
      case _FPGA_TRUNC_           : strcpy(name, "trunc_"); break;
      case _FPGA_ADD              : strcpy(name, "add"); break;
      case _FPGA_INC              : strcpy(name, "inc"); break;
      case _FPGA_MULT2D           : strcpy(name, "mult2D"); break;
      case _FPGA_EL_DIV           : strcpy(name, "el_div"); break;
      case _FPGA_EL_MULT          : strcpy(name, "el_mult"); break;
      case _FPGA_SIGN2            : strcpy(name, "sign2"); break;
      case _FPGA_SUM2D_ROWWISE    : strcpy(name, "sum2D_rowwise"); break;
      case _FPGA_SUM2D_COLWISE    : strcpy(name, "sum2D_colwise"); break;
      case _FPGA_MAX              : strcpy(name, "max"); break;
      case _FPGA_MIN              : strcpy(name, "min"); break;
      case _FPGA_SUM              : strcpy(name, "sum"); break;
      case _FPGA_SUM_ABS          : strcpy(name, "sum_abs"); break;
      case _FPGA_REDUCE           : strcpy(name, "reduce"); break;
      case _FPGA_REDUCE_OP        : strcpy(name, "reduce_op"); break;
      case _FPGA_REDUCE_SUM2D     : strcpy(name, "reduce_sum2D"); break;
      case _FPGA_REDUCTION        : strcpy(name, "reduction"); break;
      case _FPGA_REDUCTION_BACK   : strcpy(name, "reduction_back"); break;
      case _FPGA_RELU             : strcpy(name, "relu"); break;
      case _FPGA_D_RELU           : strcpy(name, "d_relu"); break;
      case _FPGA_THRESHOLDED_RELU  : strcpy(name, "thresholded_relu"); break;
      case _FPGA_D_THRESHOLDED_RELU  : strcpy(name, "d_thresholded_relu"); break;
      case _FPGA_LEAKY_RELU         : strcpy(name, "leaky_relu"); break;
      case _FPGA_D_LEAKY_RELU        : strcpy(name, "d_leaky_relu"); break;
      case _FPGA_ELU                 : strcpy(name, "elu"); break;
      case _FPGA_D_ELU               : strcpy(name, "d_elu"); break;
      case _FPGA_SOFTPLUS            : strcpy(name, "softplus"); break;
      case _FPGA_D_SOFTPLUS          : strcpy(name, "d_softplus"); break;
      case _FPGA_SOFTSIGN            : strcpy(name, "softsign"); break;
      case _FPGA_D_SOFTSIGN          : strcpy(name, "d_softsign"); break;
      case _FPGA_LINEAR              : strcpy(name, "linear"); break;
      case _FPGA_D_LINEAR            : strcpy(name, "d_linear"); break;
      case _FPGA_SIGMOID             : strcpy(name, "sigmoid"); break;
      case _FPGA_D_SIGMOID           : strcpy(name, "d_sigmoid"); break;
      case _FPGA_HARD_SIGMOID        : strcpy(name, "hard_sigmoid"); break;
      case _FPGA_D_HARD_SIGMOID      : strcpy(name, "d_hard_sigmoid"); break;
      case _FPGA_EXP                 : strcpy(name, "exp"); break;
      case _FPGA_D_EXP               : strcpy(name, "d_exp"); break;
      case _FPGA_TANH                : strcpy(name, "tanh"); break;
      case _FPGA_D_TANH              : strcpy(name, "d_tanh"); break;
      case _FPGA_SOFTMAX             : strcpy(name, "softmax"); break;
      case _FPGA_D_SOFTMAX           : strcpy(name, "d_softmax"); break;
      case _FPGA_PERMUTE_CHANELS_LAST  : strcpy(name, "permute_channels_last"); break;
      case _FPGA_PERMUTE_CHANELS_FIRST  : strcpy(name, "permute_channels_first"); break;
      case _FPGA_PERMUTE_BATCH_LAST     : strcpy(name, "permute_batch_last"); break;
      case _FPGA_PERMUTE_BATCH_FIRST    : strcpy(name, "permute_batch_first"); break;
      case _FPGA_IM2COL                 : strcpy(name, "im2col"); break;
      case _FPGA_CONV2D                 : strcpy(name, "conv2d"); break;
      case _FPGA_CONV2D_GRAD            : strcpy(name, "conv2d_grad"); break;
      case _FPGA_CONV2D_BACK            : strcpy(name, "conv2d_back"); break;
      case _FPGA_CENT                   : strcpy(name, "cent"); break;
      case _FPGA_ACCURACY               : strcpy(name, "accuracy"); break;
      case _FPGA_MPOOL2D               : strcpy(name, "mpool2d"); break;
      case _FPGA_MPOOL2D_BACK           : strcpy(name, "mpool2d_back"); break;
      case _FPGA_AVGPOOL2D              : strcpy(name, "avgpool2d"); break;
      case _FPGA_AVGPOOL2D_BACK         : strcpy(name, "avgpool2d_back"); break;
      case _FPGA_REPEAT_NN              : strcpy(name, "repeat_nn"); break;
      case _FPGA_D_REPEAT_NN            : strcpy(name, "d_repeat_nn"); break;
      default                          : strcpy(name, "?????"); break;
  }
}

struct timeval time_ini_fpga[_NUM_FPGA_FUNCS];
unsigned long long acc_time_fpga[_NUM_FPGA_FUNCS];

void _profile_fpga(int f_id, int end) {
  num_instances_fpga[f_id]++;
  if (!end) gettimeofday(&time_ini_fpga[f_id], NULL);
  else {
      timeval t1;
      gettimeofday(&t1, NULL);
      acc_time_fpga[f_id] += ((t1.tv_sec - time_ini_fpga[f_id].tv_sec) * 1000000) +
                        (t1.tv_usec - time_ini_fpga[f_id].tv_usec);
  }
}

void _show_profile_fpga() {
  printf("\nFPGA functions called:\n");
  for (int i=0; i<_NUM_FPGA_FUNCS; i++) {
    if (num_instances_fpga[i] != 0) {
      char func_name[50];
      _profile_fpga_funcname(i, func_name);
      printf("%-50s: %d instances, %llu us\n", func_name, num_instances_fpga[i], acc_time_fpga[i]);
    }
  }
  printf("Memory: %f MB\n", mb_memory_needed_fpga);
}

void _profile_fpga_add_tensor(int size) {
  mb_memory_needed_fpga += (float)size / 1024.0 / 1024.0;
  printf("accumulated tensor memory (fpga): size %f\n", mb_memory_needed_fpga);
}

void _profile_fpga_remove_tensor(int size) {
  mb_memory_needed_fpga -= (float)size / 1024.0 / 1024.0;
}


// FPGA initialization and finalization ----------------------
//
void fpga_init(){ // initialize only once

    cl_int err;
    std::string binaryFile = "eddl-gemx.xclbin";
    unsigned fileBufSize;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};

    devices.resize(1);
    OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));
    /*OCL_CHECK(err, tensor_op= cl::Kernel(program,"tensor_op", &err));*/
    OCL_CHECK(err, multitensor_op = cl::Kernel(program,"k_multitensor_op", &err));
    OCL_CHECK(err, kernel_add = cl::Kernel(program,"k_add", &err));
    //OCL_CHECK(err, mult2D = cl::Kernel(program,"k_mult2D", &err));
    OCL_CHECK(err, sum2D_rowwise = cl::Kernel(program,"k_sum2D_rowwise", &err));
    OCL_CHECK(err, kernel_cent = cl::Kernel(program,"k_cent", &err));
    OCL_CHECK(err, relu_soft_d = cl::Kernel(program,"k_relu_soft_d", &err));
    OCL_CHECK(err, reduce_sum2D = cl::Kernel(program,"k_reduce_sum2D", &err));
    OCL_CHECK(err, kernel_core = cl::Kernel(program,"k_core", &err));
    OCL_CHECK(err, kernel_accuracy = cl::Kernel(program,"k_accuracy", &err));
    OCL_CHECK(err, kernel_total_sum = cl::Kernel(program,"k_total_sum", &err));
    //OCL_CHECK(err, el_div = cl::Kernel(program,"k_el_div", &err));
    //OCL_CHECK(err, kernel_normalize = cl::Kernel(program,"k_normalize", &err));*/
    //kernel_gemx = clCreateKernel(program(), "gemxKernel_0", &err);
    if (err != CL_SUCCESS) printf("Error creating kernel\n");

}

void close_fpga(){
 //delete fileBuf;
}


// ----------------------------------------------
// Tensor creation and delete operations
//
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

// ---------------------------------------------------
// Copy operations
//

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
    //OCL_CHECK(err, err= q.enqueueWriteBuffer((A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(float), nptr, nullptr, &blocking_event));
    OCL_CHECK(err, err= q.enqueueWriteBuffer((A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(float), nptr, nullptr, &blocking_event));
    //printf("A->sizeof(float): %f\n", A->size*sizeof(float));
    //printf("nptr-> %f\n", nptr);
    //printf("A->: %f\n", A->fpga_ptr);
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


// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_transpose       = 1;
char fpga_set_cpuemu_copy            = 1;
char fpga_set_cpuemu_fill_           = 1;
char fpga_set_cpuemu_fill            = 1;
char fpga_set_cpuemu_select          = 1;
char fpga_set_cpuemu_select_back     = 1;
char fpga_set_cpuemu_set_select      = 1;
char fpga_set_cpuemu_set_select_back = 1;
char fpga_set_cpuemu_select2         = 1;
char fpga_set_cpuemu_deselect        = 1;
char fpga_set_cpuemu_concat          = 1;

// ---------------------------------------------------
// Support functions

// -----------------------------------------------------------------
// all
//
void fpga_cpuemu_transpose(Tensor *A, Tensor *B) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_transpose(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_transpose(Tensor * A, Tensor * B) {
    _profile_fpga(_FPGA_TRANSPOSE, 0);
    if (fpga_set_cpuemu_transpose == 1) {
        fpga_cpuemu_transpose(A, B);
    } else {
        printf("fpga_transpose not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_TRANSPOSE, 1);
}

// -----------------------------------------------------------------
// copy
//
void fpga_cpuemu_copy(Tensor *A, Tensor *B) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_copy(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_copy(Tensor * A, Tensor * B){
    _profile_fpga(_FPGA_COPY, 0);
    if (fpga_set_cpuemu_copy == 1) {
        fpga_cpuemu_copy(A, B);
    } else {
      int Asize = A->size * sizeof(float);
      if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
      fpga_copy_from_fpga(A, A->ptr);
      fpga_copy_to_fpga(A->ptr, B);
    }
    _profile_fpga(_FPGA_COPY, 1);
}

// -----------------------------------------------------------------
// fill_
//
void fpga_cpuemu_fill_(Tensor *A, float v) {
  int Asize = A->size * sizeof(float);
  cpu_fill_(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_fill_(Tensor *A, float v){
    _profile_fpga(_FPGA_FILL_, 0);
    if (fpga_set_cpuemu_fill_ == 1) {
        fpga_cpuemu_fill_(A, v);
    } else {
        printf("fpga_fill_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_FILL_, 1);
}

// -----------------------------------------------------------------
// fill
//
void fpga_cpuemu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_fill(A, aini, aend, B, bini, bend, inc);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_fill(Tensor * A, int aini, int aend, Tensor * B, int bini, int bend, int inc){
    _profile_fpga(_FPGA_FILL, 0);
    if (fpga_set_cpuemu_fill == 1) {
        fpga_cpuemu_fill(A, aini, aend, B, bini, bend, inc);
    } else {
        printf("fpga_fill not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_FILL, 1);
}

// -----------------------------------------------------------------
// select
//
void fpga_cpuemu_select(Tensor *A, Tensor *B, SelDescriptor *sd) {
    printf("fpga_cpuemu_select not implemented yet\n");
    exit(1);
}

void fpga_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile_fpga(_FPGA_SELECT, 0);
    if (fpga_set_cpuemu_transpose == 1) {
        fpga_cpuemu_select(A, B, sd);
    } else {
        printf("fpga_select not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SELECT, 1);
}

// -----------------------------------------------------------------
// select_back
//
void fpga_cpuemu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd) {
    printf("fpga_cpuemu_select_back not implemented yet\n");
    exit(1);
}

void fpga_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile_fpga(_FPGA_SELECT_BACK, 0);
    if (fpga_set_cpuemu_select_back == 1) {
        fpga_cpuemu_select_back(A, B, sd);
    } else {
        printf("fpga_select_back not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SELECT_BACK, 1);
}

// -----------------------------------------------------------------
// set_select
//
void fpga_cpuemu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd) {
    printf("fpga_cpuemu_set_select not implemented yet\n");
    exit(1);
}

void fpga_set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile_fpga(_FPGA_SET_SELECT, 0);
    if (fpga_set_cpuemu_select == 1) {
        fpga_cpuemu_set_select(A, B, sd);
    } else {
        printf("fpga_transpose not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SET_SELECT, 1);
}

// -----------------------------------------------------------------
// set_select_back
//
void fpga_cpuemu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd) {
    printf("fpga_cpuemu_set_select_back not implemented yet\n");
    exit(1);
}

void fpga_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile_fpga(_FPGA_SET_SELECT_BACK, 0);
    if (fpga_set_cpuemu_set_select_back == 1) {
        fpga_cpuemu_set_select_back(A, B, sd);
    } else {
        printf("fpga_set_select_back not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SET_SELECT_BACK, 1);
}

// -----------------------------------------------------------------
// select2
//
void fpga_cpuemu_select(Tensor * A, Tensor * B, vector<int> sind, int ini, int end,bool mask_zeros) {
    printf("fpga_cpuemu_select(2) not implemented yet\n");
    exit(1);
}

void fpga_select(Tensor * A, Tensor * B, vector<int> sind, int ini, int end,bool mask_zeros){
    _profile_fpga(_FPGA_SELECT2, 0);
    if (fpga_set_cpuemu_select2 == 1) {
        fpga_cpuemu_select(A, B, sind, ini, end, mask_zeros);
    } else {
        printf("fpga_select(2) not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SELECT2, 1);
}

// -----------------------------------------------------------------
// deselect
//
void fpga_cpuemu_deselect(Tensor * A, Tensor * B, vector<int> sind, int ini, int end, int inc, bool mask_zeros) {
    printf("fpga_cpuemu_deslect not implemented yet\n");
    exit(1);
}

void fpga_deselect(Tensor * A, Tensor * B, vector<int> sind, int ini, int end,int inc,bool mask_zeros){
    _profile_fpga(_FPGA_DESELECT, 0);
    if (fpga_set_cpuemu_deselect == 1) {
        fpga_cpuemu_deselect(A, B, sind, ini, end, inc, mask_zeros);
    } else {
        printf("fpga_deselect not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_DESELECT, 1);
}

// -----------------------------------------------------------------
// concat
//
void fpga_cpuemu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative) {
    printf("fpga_cpuemu_concat not implemented yet\n");
    exit(1);
}

void fpga_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative){
    _profile_fpga(_FPGA_CONCAT, 0);
    if (fpga_set_cpuemu_concat == 1) {
        fpga_cpuemu_concat(A, t, axis, derivative);
    } else {
        printf("fpga_concat not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_CONCAT, 1);
}
