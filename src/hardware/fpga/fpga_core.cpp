/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include "eddl/hardware/fpga/xcl2.hpp"
#include <vector>
#include <math.h>
#include <float.h>
#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"
#include "eddl/hardware/fpga/fpga_hw.h"
#include <sys/time.h>
#include "eddl/hardware/cpu/cpu_tensor.h"

int next_fpga_tensor_id = 1;
int num_tensors_created = 0;

#define MAX_BUFFER_POOL 10000

cl::Buffer *fpga_ptr_buffer_pool[MAX_BUFFER_POOL];
long fpga_size_buffer_pool[MAX_BUFFER_POOL];
int fpga_inuse_buffer_pool[MAX_BUFFER_POOL];
int fpga_free_buffer_pool[MAX_BUFFER_POOL];
int fpga_num_buffer_pool_slots;

cl::Context      context;
cl::CommandQueue q;
cl::CommandQueue com;
cl::Program      program;

// activation kernels (22)
cl::Kernel kernel_relu,   kernel_d_relu,  kernel_thresholded_relu,    kernel_d_thresholded_relu, kernel_leaky_relu,     kernel_d_leaky_relu;
cl::Kernel kernel_elu,    kernel_d_elu,   kernel_softplus,            kernel_d_softplus,         kernel_softsign,       kernel_d_softsign;
cl::Kernel kernel_linear, kernel_d_linear,kernel_sigmoid,             kernel_d_sigmoid,          kernel_hard_sigmoid,   kernel_d_hard_sigmoid;
cl::Kernel kernel_exp,    kernel_d_exp,   kernel_tanh, kernel_d_tanh, kernel_softmax,            kernel_d_softmax;

// bn kernels (4)
cl::Kernel kernel_permute_channels_last, kernel_permute_channels_first;
cl::Kernel kernel_permute_batch_last,    kernel_permute_batch_first;

// comparison kernels (20)
cl::Kernel kernel_all,         kernel_any,        kernel_isfinite,    kernel_isinf;
cl::Kernel kernel_isnan,       kernel_isneginf,   kernel_isposinf,    kernel_equal2;
cl::Kernel kernel_logical_and, kernel_logical_or, kernel_logical_not, kernel_logical_xor;
cl::Kernel kernel_allclose,    kernel_isclose,    kernel_greater,     kernel_greater_equal;
cl::Kernel kernel_less,        kernel_less_equal, kernel_equal,       kernel_not_equal;
cl::Kernel kernel_greater_vector, kernel_greater_equal_vector, kernel_less_vector;
cl::Kernel kernel_less_equal_vector, kernel_equal_vector, kernel_not_equal_vector;

// core kernels (11)
cl::Kernel kernel_transpose,   kernel_copy,        kernel_fill_,      kernel_fill;
cl::Kernel kernel_select,      kernel_select_back, kernel_set_select, kernel_set_select_back;
cl::Kernel kernel_set_select2, kernel_deselect,    kernel_concat;
cl::Kernel kernel_select_nn,   kernel_select_back_nn, kernel_set_select_nn, kernel_set_select_back_nn;

// conv kernels (2)
cl::Kernel kernel_im2col,      kernel_conv2d;

// create kernels (3)
cl::Kernel kernel_range, kernel_eye, kernel_diag;

// da kernels (6)
cl::Kernel kernel_single_shift, kernel_single_rotate, kernel_single_scale;
cl::Kernel kernel_single_flip,  kernel_single_crop;
cl::Kernel kernel_crop_scale_random;

// generator kernels (4)
cl::Kernel kernel_rand_uniform, kernel_signed_uniform, kernel_rand_binary, kernel_rand_normal;

// losses kernels (1)
cl::Kernel kernel_cent;

// metrics kernels (22)
cl::Kernel kernel_accuracy, kernel_bin_accuracy;

// pool kernels (4)
cl::Kernel kernel_mpool2D, kernel_mpool2D_back, kernel_avgpool2D, kernel_avgpool2D_back;

// reduction kernels (5)
cl::Kernel kernel_reduce, kernel_reduce_op, kernel_reduce_sum2D, kernel_reduction, kernel_reduction_back;

// tensor_nn kernels (2)
cl::Kernel kernel_repeat_nn, kernel_d_repeat_nn;

// math kernels (46)
cl::Kernel kernel_abs,       kernel_acos,   kernel_add,      kernel_asin,       kernel_atan,          kernel_ceil,          kernel_clamp;
cl::Kernel kernel_cos,       kernel_cosh,   kernel_mod,      kernel_mult,       kernel_trunc,         kernel_sum_abs;
cl::Kernel kernel_floor,     kernel_inv,    kernel_log,      kernel_log2,       kernel_log10,         kernel_logn;
cl::Kernel kernel_normalize, kernel_pow,    kernel_powb,     kernel_reciprocal, kernel_remainder,     kernel_round,         kernel_rsqrt;
cl::Kernel kernel_sign,      kernel_sin,    kernel_sinh,     kernel_sqr,        kernel_sqrt,          kernel_tan;
cl::Kernel kernel_inc,       kernel_el_div, kernel_el_mult,  kernel_sign2,      kernel_sum2D_rowwise, kernel_sum2D_colwise;
cl::Kernel kernel_max,       kernel_min,    kernel_sum,      kernel_mult2d,     kernel_add_2;
cl::Kernel kernel_maximum,   kernel_maximum_float, kernel_minimum, kernel_minimum_float;


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
      case _FPGA_ABS             : strcpy(name, "abs"); break;
      case _FPGA_ACOS            : strcpy(name, "acos"); break;
      case _FPGA_ASIN            : strcpy(name, "asin"); break;
      case _FPGA_ATAN            : strcpy(name, "atan"); break;
      case _FPGA_CEIL            : strcpy(name, "ceil"); break;
      case _FPGA_CLAMP           : strcpy(name, "clamp"); break;
      case _FPGA_COS             : strcpy(name, "cos"); break;
      case _FPGA_COSH            : strcpy(name, "cosh"); break;
      case _FPGA_FLOOR           : strcpy(name, "floor"); break;
      case _FPGA_INV             : strcpy(name, "inv"); break;
      case _FPGA_LOG             : strcpy(name, "log"); break;
      case _FPGA_LOG2            : strcpy(name, "log2"); break;
      case _FPGA_LOG10           : strcpy(name, "log10"); break;
      case _FPGA_LOGN            : strcpy(name, "logn"); break;
      case _FPGA_MOD             : strcpy(name, "mod"); break;
      case _FPGA_MULT            : strcpy(name, "mult"); break;
      case _FPGA_NORMALIZE       : strcpy(name, "normalize"); break;
      case _FPGA_POW             : strcpy(name, "pow"); break;
      case _FPGA_POWB            : strcpy(name, "powb"); break;
      case _FPGA_RECIPROCAL      : strcpy(name, "reciprocal"); break;
      case _FPGA_REMAINDER       : strcpy(name, "remainder"); break;
      case _FPGA_ROUND           : strcpy(name, "round"); break;
      case _FPGA_RSQRT           : strcpy(name, "rsqrt"); break;
      case _FPGA_SIGN            : strcpy(name, "sign"); break;
      case _FPGA_SIN             : strcpy(name, "sin"); break;
      case _FPGA_SINH            : strcpy(name, "sinh"); break;
      case _FPGA_SQR             : strcpy(name, "sqr"); break;
      case _FPGA_SQRT            : strcpy(name, "sqrt"); break;
      case _FPGA_TAN             : strcpy(name, "tan"); break;
      case _FPGA_TRUNC           : strcpy(name, "trunc"); break;
      case _FPGA_ADD              : strcpy(name, "add"); break;
      case _FPGA_ADD_2            : strcpy(name, "add_2"); break;
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
      case _FPGA_PROD                   : strcpy(name, "prod"); break;
      case _FPGA_PROD_2                 : strcpy(name, "prod_2"); break;
      case _FPGA_SUM_2                  : strcpy(name, "sum_2"); break;
      case _FPGA_MAXIMUM                : strcpy(name, "maximum"); break;
      case _FPGA_MAXIMUM_FLOAT          : strcpy(name, "maximum_float"); break;
      case _FPGA_MINIMUM                : strcpy(name, "minimum"); break;
      case _FPGA_MINIMUM_FLOAT          : strcpy(name, "minimum_float"); break;
      case _FPGA_ARGMIN                 : strcpy(name, "argmin"); break;
      case _FPGA_ARGMAX                 : strcpy(name, "argmax"); break;
      case _FPGA_VAR                    : strcpy(name, "var"); break;
      case _FPGA_VAR_2                  : strcpy(name, "var_2"); break;
      case _FPGA_STD                    : strcpy(name, "std"); break;
      case _FPGA_STD_2                  : strcpy(name, "std_2"); break;
      case _FPGA_MEDIAN                 : strcpy(name, "median"); break;
      case _FPGA_MEDIAN_2               : strcpy(name, "median_2"); break;
      case _FPGA_SORT                   : strcpy(name, "sort"); break;
      case _FPGA_ARGSORT                : strcpy(name, "argsort"); break;

      default                          : strcpy(name, "?????"); break;
  }
}

struct timeval time_ini_fpga[_NUM_FPGA_FUNCS];
unsigned long long acc_time_fpga[_NUM_FPGA_FUNCS];

void _profile_fpga(int f_id, int end) {
#ifdef FPGA_DEBUG
  char func_name[50];
  _profile_fpga_funcname(f_id, func_name);
  if (!end) printf("%s\n", func_name);
  if (end) printf("\n");
#endif
  num_instances_fpga[f_id]++;
  if (!end) gettimeofday(&time_ini_fpga[f_id], NULL);
  else {
      timeval t1;
      gettimeofday(&t1, NULL);
      acc_time_fpga[f_id] += ((t1.tv_sec - time_ini_fpga[f_id].tv_sec) * 1000000) +
                        (t1.tv_usec - time_ini_fpga[f_id].tv_usec);
  }
}

void _profile_fpga_tensor(Tensor *T) {
#ifdef FPGA_DEBUG
  // We read the tensor from FPGA
  fpga_copy_from_fpga(T, T->ptr);
  float min = FLT_MAX;
  float max = FLT_MIN;
  float sum = 0.f;
  float avg;
  for (int i=0; i<T->size; i++) {
    if (T->ptr[i] > max) max = T->ptr[i];
    if (T->ptr[i] < min) min = T->ptr[i];
    sum += T->ptr[i];
  }
  avg = sum / (float)T->size;
  printf("  - Tensor id %d size %d size_fpga %d shape0 %d shape1 %d (cpu_ptr %p). Min %8.4f Max %8.4f Avg %8.4f\n", T->fpga_tensor_id, T->size, T->fpga_size, T->shape[0], T->shape[1], T->ptr, min, max, avg);
#endif
}

void _show_profile_fpga() {
#ifdef FPGA_SHOW_PROFILE
  printf("\n---------------------------------------\nFPGA functions called:\n");
  for (int i=0; i<_NUM_FPGA_FUNCS; i++) {
    if (num_instances_fpga[i] != 0) {
      char func_name[50];
      _profile_fpga_funcname(i, func_name);
      printf("%-50s: %d instances, %llu us\n", func_name, num_instances_fpga[i], acc_time_fpga[i]);
    }
  }
  printf("Memory: %f MB\n", mb_memory_needed_fpga);
  printf("---------------------------------------\n");
#endif
}

void _profile_fpga_add_tensor(int size) {
//  printf("tensor add: size in MB: %6.4f\n", (float)size / 1024.0 / 1024.0);
  mb_memory_needed_fpga += (float)size / 1024.0 / 1024.0;
  num_tensors_created++;
//  printf("tensor add: size in MB: %6.4f (active tensors %d)\n", (float)size / 1024.0 / 1024.0, num_tensors_created);
#ifdef FPGA_DEBUG
  printf("    (accumulated tensor memory %f MB)\n", mb_memory_needed_fpga);
#endif
}

void _profile_fpga_remove_tensor(int size) {
  mb_memory_needed_fpga -= (float)size / 1024.0 / 1024.0;
  num_tensors_created--;
}



// FPGA initialization and finalization ----------------------
//
void fpga_init(){ // initialize only once

    cl_int err;
    std::string binaryFile = "eddl.xclbin";
    unsigned fileBufSize;

    std::vector<cl::Device> devices = xcl::get_xil_devices();

    cl::Device device = devices[0];
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};

    devices.resize(1);
    OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));

    #ifdef K_ENABLED_RELU
    OCL_CHECK(err, kernel_relu = cl::Kernel(program,"k_relu", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_RELU
    OCL_CHECK(err, kernel_d_relu = cl::Kernel(program,"k_d_relu", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_THRESHOLDED_RELU
    OCL_CHECK(err, kernel_thresholded_relu = cl::Kernel(program,"k_thresholded_relu", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_THRESHOLDED_RELU
    OCL_CHECK(err, kernel_d_thresholded_relu = cl::Kernel(program,"k_d_thresholded_relu", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LEAKY_RELU
    OCL_CHECK(err, kernel_leaky_relu = cl::Kernel(program,"k_leaky_relu", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_LEAKY_RELU
    OCL_CHECK(err, kernel_d_leaky_relu = cl::Kernel(program,"k_d_leaky_relu", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ELU
    OCL_CHECK(err, kernel_elu = cl::Kernel(program,"k_elu", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_ELU
    OCL_CHECK(err, kernel_d_elu = cl::Kernel(program,"k_d_elu", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SOFTPLUS
    OCL_CHECK(err, kernel_softplus = cl::Kernel(program,"k_softplus", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_SOFTPLUS
    OCL_CHECK(err, kernel_d_softplus = cl::Kernel(program,"k_d_softplus", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SOFTSIGN
    OCL_CHECK(err, kernel_softsign = cl::Kernel(program,"k_softsign", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_SOFTPLUS
    OCL_CHECK(err, kernel_d_softsign = cl::Kernel(program,"k_d_softsign", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LINEAR
    OCL_CHECK(err, kernel_linear = cl::Kernel(program,"k_linear", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_LINEAR
    OCL_CHECK(err, kernel_d_linear = cl::Kernel(program,"k_d_linear", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_LINEAR
    OCL_CHECK(err, kernel_d_softplus = cl::Kernel(program,"k_d_linear", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SIGMOID
    OCL_CHECK(err, kernel_sigmoid = cl::Kernel(program,"k_sigmoid", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_SIGMOID
    OCL_CHECK(err, kernel_d_sigmoid = cl::Kernel(program,"k_d_sigmoid", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_HARD_SIGMOID
    OCL_CHECK(err, kernel_hard_sigmoid = cl::Kernel(program,"k_hard_sigmoid", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_HARD_SIGMOID
    OCL_CHECK(err, kernel_d_hard_sigmoid = cl::Kernel(program,"k_d_hard_sigmoid", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_EXP
    OCL_CHECK(err, kernel_exp = cl::Kernel(program,"k_exp", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_EXP
    OCL_CHECK(err, kernel_d_exp = cl::Kernel(program,"k_d_exp", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_TANH
    OCL_CHECK(err, kernel_tanh = cl::Kernel(program,"k_tanh", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_TANH
    OCL_CHECK(err, kernel_d_tanh = cl::Kernel(program,"k_d_tanh", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SOFTMAX
    OCL_CHECK(err, kernel_softmax = cl::Kernel(program,"k_softmax", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_SOFTMAX
    OCL_CHECK(err, kernel_d_softmax = cl::Kernel(program,"k_d_softmax", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_PERMUTE_CHANNELS_LAST
    OCL_CHECK(err, kernel_permute_channels_last = cl::Kernel(program,"k_permute_channels_last", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_PERMUTE_CHANNELS_FIRST
    OCL_CHECK(err, kernel_permute_channels_first = cl::Kernel(program,"k_permute_channels_first", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_PERMUTE_BATCH_LAST
    OCL_CHECK(err, kernel_permute_batch_last = cl::Kernel(program,"k_permute_batch_last", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_PERMUTE_BATCH_FIRST
    OCL_CHECK(err, kernel_permute_batch_first = cl::Kernel(program,"k_permute_batch_first", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ALL
    OCL_CHECK(err, kernel_all = cl::Kernel(program,"k_all", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ANY
    OCL_CHECK(err, kernel_any = cl::Kernel(program,"k_any", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ISFINITE
    OCL_CHECK(err, kernel_isfinite = cl::Kernel(program,"k_isfinite", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ISINF
    OCL_CHECK(err, kernel_isinf = cl::Kernel(program,"k_isinf", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ISNAN
    OCL_CHECK(err, kernel_isnan = cl::Kernel(program,"k_isnan", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ISNEGINF
    OCL_CHECK(err, kernel_isneginf = cl::Kernel(program,"k_isneginf", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ISPOSINF
    OCL_CHECK(err, kernel_isposinf = cl::Kernel(program,"k_isposinf", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LOGICAL_AND
    OCL_CHECK(err, kernel_logical_and = cl::Kernel(program,"k_logical_and", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LOGICAL_OR
    OCL_CHECK(err, kernel_logical_or = cl::Kernel(program,"k_logical_or", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LOGICAL_NOT
    OCL_CHECK(err, kernel_logical_not = cl::Kernel(program,"k_logical_not", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LOGICAL_XOR
    OCL_CHECK(err, kernel_logical_xor = cl::Kernel(program,"k_logical_xor", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ALLCLOSE
    OCL_CHECK(err, kernel_allclose = cl::Kernel(program,"k_allclose", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ISCLOSE
    OCL_CHECK(err, kernel_isclose = cl::Kernel(program,"k_isclose", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_GREATER
    OCL_CHECK(err, kernel_greater = cl::Kernel(program,"k_greater", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_GREATER_EQUAL
    OCL_CHECK(err, kernel_greater_equal = cl::Kernel(program,"k_greater_equal", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LESS
    OCL_CHECK(err, kernel_less = cl::Kernel(program,"k_less", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LESS_EQUAL
    OCL_CHECK(err, kernel_less_equal = cl::Kernel(program,"k_less_equal", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_EQUAL
    OCL_CHECK(err, kernel_equal = cl::Kernel(program,"k_equal", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_NOT_EQUAL
    OCL_CHECK(err, kernel_not_equal = cl::Kernel(program,"k_not_equal", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif

   #ifdef K_ENABLED_GREATER_VECTOR
    OCL_CHECK(err, kernel_greater_vector = cl::Kernel(program,"k_greater_vector", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_GREATER_EQUAL_VECTOR
    OCL_CHECK(err, kernel_greater_equal_vector = cl::Kernel(program,"k_greater_equal_vecotr", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LESS_VECTOR
    OCL_CHECK(err, kernel_less_vector = cl::Kernel(program,"k_less_vector", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LESS_EQUAL_VECTOR
    OCL_CHECK(err, kernel_less_equal_vector = cl::Kernel(program,"k_less_equal_vector", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_EQUAL_VECTOR
    OCL_CHECK(err, kernel_equal_vector = cl::Kernel(program,"k_equal_vector", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_NOT_EQUAL_VECTOR
    OCL_CHECK(err, kernel_not_equal_vector = cl::Kernel(program,"k_not_equal_vector", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif


    #ifdef K_ENABLED_EQUAL2
    OCL_CHECK(err, kernel_equal2 = cl::Kernel(program,"k_equal2", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_TRANSPOSE
    OCL_CHECK(err, kernel_transpose = cl::Kernel(program,"k_transpose", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_COPY
    OCL_CHECK(err, kernel_copy = cl::Kernel(program,"k_copy", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_FILL_
    OCL_CHECK(err, kernel_fill_ = cl::Kernel(program,"k_fill_", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_FILL
    OCL_CHECK(err, kernel_fill = cl::Kernel(program,"k_fill", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SELECT
    OCL_CHECK(err, kernel_select = cl::Kernel(program,"k_select", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SELECT_BACK
    OCL_CHECK(err, kernel_select_back = cl::Kernel(program,"k_select_back", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SET_SELECT
    OCL_CHECK(err, kernel_set_select = cl::Kernel(program,"k_set_select", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SET_SELECT_BACK
    OCL_CHECK(err, kernel_set_select_back = cl::Kernel(program,"k_set_select_back", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif

    #ifdef K_ENABLED_SELECT_NN
    OCL_CHECK(err, kernel_select_nn = cl::Kernel(program,"k_select_nn", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SELECT_BACK_NN
    OCL_CHECK(err, kernel_select_back = cl::Kernel(program,"k_select_back_nn", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SET_SELECT_NN
    OCL_CHECK(err, kernel_set_select_nn = cl::Kernel(program,"k_set_select_nn", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SET_SELECT_BACK_NN
    OCL_CHECK(err, kernel_set_select_back_nn = cl::Kernel(program,"k_set_select_back_nn", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif

    #ifdef K_ENABLED_SET_SELECT2
    OCL_CHECK(err, kernel_set_select2 = cl::Kernel(program,"k_set_select2", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_DESELECT
    OCL_CHECK(err, kernel_deselect = cl::Kernel(program,"k_deselect", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_CONCAT
    OCL_CHECK(err, kernel_concat = cl::Kernel(program,"k_concat", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_IM2COL
    OCL_CHECK(err, kernel_im2col = cl::Kernel(program,"k_im2col", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_CONV2D
    OCL_CHECK(err, kernel_conv2d = cl::Kernel(program,"k_conv2d", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_RANGE
    OCL_CHECK(err, kernel_range = cl::Kernel(program,"k_range", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_EYE
    OCL_CHECK(err, kernel_eye = cl::Kernel(program,"k_eye", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_DIAG
    OCL_CHECK(err, kernel_diag = cl::Kernel(program,"k_diag", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SINGLE_SHIFT
    OCL_CHECK(err, kernel_single_shift = cl::Kernel(program,"k_single_shift", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SINGLE_ROTATE
    OCL_CHECK(err, kernel_single_rotate = cl::Kernel(program,"k_single_rotate", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SINGLE_SCALE
    OCL_CHECK(err, kernel_single_scale = cl::Kernel(program,"k_single_scale", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SINGLE_FLIP
    OCL_CHECK(err, kernel_single_flip = cl::Kernel(program,"k_single_flip", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SINGLE_CROP
    OCL_CHECK(err, kernel_single_crop = cl::Kernel(program,"k_single_crop", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_CROP_SCALE_RANDOM
    OCL_CHECK(err, kernel_crop_scale_random = cl::Kernel(program,"k_crop_scale_random", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_RAND_UNIFORM
    OCL_CHECK(err, kernel_rand_uniform = cl::Kernel(program,"k_single_rand_uniform", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_RAND_SIGNED_UNIFORM
    OCL_CHECK(err, kernel_signed_uniform = cl::Kernel(program,"k_single_signed_uniform", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_RAND_BINARY
    OCL_CHECK(err, kernel_rand_binary = cl::Kernel(program,"k_single_rand_binary", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_RAND_NORMAL
    OCL_CHECK(err, kernel_rand_normal = cl::Kernel(program,"k_single_rand_normal", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_CENT
    OCL_CHECK(err, kernel_cent = cl::Kernel(program,"k_cent", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ACCURACY
    OCL_CHECK(err, kernel_accuracy = cl::Kernel(program,"k_accuracy", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_BIN_ACCURACY
    OCL_CHECK(err, kernel_bin_accuracy = cl::Kernel(program,"k_bin_accuracy", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MPOOL2D
    OCL_CHECK(err, kernel_mpool2D = cl::Kernel(program,"k_mpool2D", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MPOOL2D_BACK
    OCL_CHECK(err, kernel_mpool2D_back = cl::Kernel(program,"k_mpool2D_back", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_AVGPOOL2D
    OCL_CHECK(err, kernel_avgpool2D = cl::Kernel(program,"k_mpool2D", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_AVGPOOL2D_BACK
    OCL_CHECK(err, kernel_avgpool2D_back = cl::Kernel(program,"k_avgpool2D_back", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_REDUCE
    OCL_CHECK(err, kernel_reduce = cl::Kernel(program,"k_reduce", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_REDUCE_OP
    OCL_CHECK(err, kernel_reduce_op = cl::Kernel(program,"k_reduce_op", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_REDUCE_SUM2D
    OCL_CHECK(err, kernel_reduce_sum2D = cl::Kernel(program,"k_reduce_sum2d", &err));
     if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_REDUCTION
    OCL_CHECK(err, kernel_reduction = cl::Kernel(program,"k_reduction", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_REDUCTION_BACK
    OCL_CHECK(err, kernel_reduction_back = cl::Kernel(program,"k_reduction_back", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_REPEAT_NN
    OCL_CHECK(err, kernel_repeat_nn = cl::Kernel(program,"k_repeat_nn", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_D_REPEAT_NN
    OCL_CHECK(err, kernel_d_repeat_nn = cl::Kernel(program,"k_d_repeat_nn", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ABS
    OCL_CHECK(err, kernel_abs = cl::Kernel(program,"k_abs", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ACOS
    OCL_CHECK(err, kernel_acos = cl::Kernel(program,"k_acos", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ADD
    OCL_CHECK(err, kernel_add = cl::Kernel(program,"k_add", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ADD_2
    OCL_CHECK(err, kernel_add_2 = cl::Kernel(program,"k_add_2", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ASIN
    OCL_CHECK(err, kernel_asin = cl::Kernel(program,"k_asin", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ATAN
    OCL_CHECK(err, kernel_atan = cl::Kernel(program,"k_atan", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_CEIL
    OCL_CHECK(err, kernel_ceil = cl::Kernel(program,"k_ceil", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_CLAMP
    OCL_CHECK(err, kernel_clamp = cl::Kernel(program,"k_clamp", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_K_COS
    OCL_CHECK(err, kernel_cos = cl::Kernel(program,"k_cos", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_COSH
    OCL_CHECK(err, kernel_cosh = cl::Kernel(program,"k_cosh", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_FLOOR
    OCL_CHECK(err, kernel_floor = cl::Kernel(program,"k_floor", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_INV
    OCL_CHECK(err, kernel_inv = cl::Kernel(program,"k_inv", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LOG
    OCL_CHECK(err, kernel_log = cl::Kernel(program,"k_log", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LOG2
    OCL_CHECK(err, kernel_log2 = cl::Kernel(program,"k_log2", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LOG10
    OCL_CHECK(err, kernel_log10 = cl::Kernel(program,"k_log10", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_LOGN
    OCL_CHECK(err, kernel_logn = cl::Kernel(program,"k_logn", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MOD
    OCL_CHECK(err, kernel_mod = cl::Kernel(program,"k_mod", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MULT
    OCL_CHECK(err, kernel_mult = cl::Kernel(program,"k_mult", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_NORMALIZE
    OCL_CHECK(err, kernel_normalize = cl::Kernel(program,"k_normalize", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_POW
    OCL_CHECK(err, kernel_pow = cl::Kernel(program,"k_pow", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_POWB
    OCL_CHECK(err, kernel_powb = cl::Kernel(program,"k_powb", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_RECIPROCAL
    OCL_CHECK(err, kernel_reciprocal = cl::Kernel(program,"k_reciprocal", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_REMAINDER
    OCL_CHECK(err, kernel_remainder = cl::Kernel(program,"k_remainder", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_ROUND
    OCL_CHECK(err, kernel_round = cl::Kernel(program,"k_round", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_RSQRT
    OCL_CHECK(err, kernel_rsqrt = cl::Kernel(program,"k_rsqrt", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SIGN
    OCL_CHECK(err, kernel_sign = cl::Kernel(program,"k_sign", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SIN
    OCL_CHECK(err, kernel_sin = cl::Kernel(program,"k_sin", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SINH
    OCL_CHECK(err, kernel_sinh = cl::Kernel(program,"k_sinh", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SQR
    OCL_CHECK(err, kernel_sqr = cl::Kernel(program,"k_sqr", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SQRT
    OCL_CHECK(err, kernel_sqrt = cl::Kernel(program,"k_sqrt", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_TAN
    OCL_CHECK(err, kernel_tan = cl::Kernel(program,"k_tan", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_TRUNC
    OCL_CHECK(err, kernel_trunc = cl::Kernel(program,"k_trunc", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_INC
    OCL_CHECK(err, kernel_inc = cl::Kernel(program,"k_inc", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MULT2D
    OCL_CHECK(err, kernel_mult2d = cl::Kernel(program,"k_mult2d", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_EL_DIV
    OCL_CHECK(err, kernel_el_div = cl::Kernel(program,"k_el_div", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_EL_MULT
    OCL_CHECK(err, kernel_el_mult = cl::Kernel(program,"k_el_mult", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SIGN2
    OCL_CHECK(err, kernel_sign2 = cl::Kernel(program,"k_sign2", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SUM2D_ROWWISE
    OCL_CHECK(err, kernel_sum2D_rowwise = cl::Kernel(program,"k_sum2d_rowwise", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SUM2D_COLWISE
    OCL_CHECK(err, kernel_sum2D_colwise = cl::Kernel(program,"k_sum2d_colwise", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MAX
    OCL_CHECK(err, kernel_max = cl::Kernel(program,"k_max", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MIN
    OCL_CHECK(err, kernel_min = cl::Kernel(program,"k_min", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SUM
    OCL_CHECK(err, kernel_sum = cl::Kernel(program,"k_sum", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_SUM_ABS
    OCL_CHECK(err, kernel_sum_abs = cl::Kernel(program,"k_sum_abs", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MAXIMUM
    OCL_CHECK(err, kernel_maximum = cl::Kernel(program,"k_maximum", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MAXIMUM_VECTOR
    OCL_CHECK(err, kernel_maximum_float = cl::Kernel(program,"k_maximum_float", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MINIMUM
    OCL_CHECK(err, kernel_minimum = cl::Kernel(program,"k_minimum", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif
    #ifdef K_ENABLED_MINIMUM_VECTOR
    OCL_CHECK(err, kernel_minimum_float = cl::Kernel(program,"k_minimum_float", &err));
    if (err != CL_SUCCESS) printf("Error creating kernel\n");
    #endif

    // Initializing buffer pool
    for (int e=0; e<MAX_BUFFER_POOL; e++) {
      fpga_ptr_buffer_pool[e] = (cl::Buffer *)nullptr;
      fpga_size_buffer_pool[e] = 0;
      fpga_inuse_buffer_pool[e] = 0;
      fpga_free_buffer_pool[e] = 1;
    }
    fpga_num_buffer_pool_slots = 0;
    //

    printf("end of fpga_init\n");
}

void close_fpga(){
 //delete fileBuf;
}


// ----------------------------------------------
// Tensor creation and delete operations
//
cl::Buffer *fpga_create_tensor(int device, int size)
{
    cl::Buffer *buffer;
    cl_int err;
#ifdef FPGA_DEBUG
    printf("    (creating tensor in fpga, size %d)\n", size);
#endif
    _profile_fpga_add_tensor(size*sizeof(float));

    // search an available slot
    int e;
    for (e=0; e<fpga_num_buffer_pool_slots; e++) {
      if (!fpga_inuse_buffer_pool[e] && !fpga_free_buffer_pool[e] & (fpga_size_buffer_pool[e] == size)) break;
    }
    if (e!=fpga_num_buffer_pool_slots) {
#ifdef FPGA_DEBUG
      printf("    reasigning buffer pool entry\n");
#endif
      fpga_inuse_buffer_pool[e] = 1;
      return fpga_ptr_buffer_pool[e];
    }
    // create a new buffer pool
    if (fpga_num_buffer_pool_slots == MAX_BUFFER_POOL) {
      printf("Error, too many buffer pools\n");
      exit(1);
    }

    // buffer pool slot creation
#ifdef FPGA_DEBUG
    printf("Creating new buffer pool entry\n");
#endif
    OCL_CHECK(err,buffer = new cl::Buffer(context,CL_MEM_READ_WRITE, size*sizeof(float), NULL, &err));
    e = fpga_num_buffer_pool_slots;
    fpga_ptr_buffer_pool[e] = buffer;
    fpga_size_buffer_pool[e] = size;
    fpga_inuse_buffer_pool[e] = 1;
    fpga_free_buffer_pool[e] = 0;
    fpga_num_buffer_pool_slots++;
    return fpga_ptr_buffer_pool[e];
}


void fpga_delete_tensor(int device, cl::Buffer *ptr, int fpga_tensor_id_p, int size)
{
#ifdef FPGA_DEBUG
    printf("    (deleting tensor in fpga, id %d)\n", fpga_tensor_id_p);
#endif

    _profile_fpga_remove_tensor(size*sizeof(float));

    // we just update the buffer pool
    //
    int e;
  //  printf("ptr to delete %p  size %d\n", ptr, size);
    for (e=0; e<fpga_num_buffer_pool_slots; e++) {
//      printf("slot %d: inuse %d free %d size %d ptr %p\n", e, fpga_inuse_buffer_pool[e], fpga_free_buffer_pool[e], fpga_size_buffer_pool[e], fpga_ptr_buffer_pool[e]);
      if (fpga_inuse_buffer_pool[e] && !fpga_free_buffer_pool[e] && (fpga_size_buffer_pool[e] == size) && (fpga_ptr_buffer_pool[e] == ptr)) break;
    }
    if (e==fpga_num_buffer_pool_slots) {
      printf("Error, delete tensor function did not find the buffer in the pool\n");
      exit(1);
    }
    fpga_inuse_buffer_pool[e] = 0;

    //delete ptr;
}

// ---------------------------------------------------
// Copy operations
//

///////////////////////////////////////////
void fpga_copy_fpga(Tensor *A, Tensor *B)
{
#ifdef FPGA_DEBUG
    printf("    (copy fpga: tensor id %d (size %d, ptr %p) -> tensor id %d (size %d, ptr %p))\n", A->fpga_tensor_id, A->size, A->fpga_ptr, B->fpga_tensor_id, B->size, B->fpga_ptr);
#endif
    cl_int err;
    cl::Event blocking_event;
    cl::Buffer *bufferA = A->fpga_ptr;
    cl::Buffer *bufferB = B->fpga_ptr;
    if (A->size > B->size) {printf("Error, copy_fpga beyond limits\n"); exit(1);}
    OCL_CHECK(err, err= q.enqueueCopyBuffer(*bufferA, *bufferB, 0, 0, A->size*sizeof(float), NULL, &blocking_event));
    q.finish();
#ifdef FPGA_DEBUG
    printf("copy completed\n");
#endif
}

void fpga_copy_to_fpga(float *nptr, Tensor *A)
{
#ifdef FPGA_DEBUG
    printf("    (copy to fpga: tensor id %d, size %d, from_cpu_ptr %p)\n", A->fpga_tensor_id, A->size, nptr);
#endif
    cl_int err;
    cl::Event blocking_event;
    cl::Buffer *buf = A->fpga_ptr;
    OCL_CHECK(err, err= q.enqueueWriteBuffer(*buf, CL_TRUE, 0, A->size*sizeof(float), nptr, nullptr, &blocking_event));
    q.finish();
}

///////////////////////////////////////////
void fpga_copy_from_fpga(Tensor *A,float *nptr)
{
#ifdef FPGA_DEBUG
    printf("    (copy from fpga: tensor id %d, size %d, to_cpu_ptr %p)\n", A->fpga_tensor_id, A->size, nptr);
#endif
    cl_int err;
    cl::Event event;
    OCL_CHECK(err, err= q.enqueueReadBuffer(*(A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(float), nptr, nullptr, &event));
    q.finish();;
}

void fpga_copy_addresses_from_fpga(SelDescriptor *SD, int size, int *nptr)
{
    cl_int err;
    cl::Event event;
    cl::Buffer *buf = SD->fpga_ptr;
    OCL_CHECK(err, err= q.enqueueReadBuffer(*buf, CL_TRUE, 0, size, nptr, nullptr, &event));
    q.finish();;
}

void fpga_destroy_memory(cl::Buffer *fpga_ptrI) {
    if (fpga_ptrI != (cl::Buffer *)nullptr) delete fpga_ptrI;
}

cl::Buffer *fpga_create_memory(long int size) {
    cl::Buffer *buffer;
    cl_int err;
    #ifdef FPGA_DEBUG
    printf("    (creating memory in fpga size %d)\n", size);
    #endif
    OCL_CHECK(err,buffer = new cl::Buffer(context,CL_MEM_READ_WRITE, size, NULL, &err));
    return buffer;
}

void fpga_copy_memory_to_fpga(void *ptr_cpu, cl::Buffer *ptr_fpga, long int size) {
#ifdef FPGA_DEBUG
    printf("    (copy memory to fpga: size %d, ptr_cpu %p)\n", size, ptr_cpu);
#endif
    cl_int err;
    cl::Event blocking_event;
    OCL_CHECK(err, err= q.enqueueWriteBuffer(*ptr_fpga, CL_TRUE, 0, size, ptr_cpu, nullptr, &blocking_event));
    q.finish();
}

void fpga_copy_memory_from_fpga(cl::Buffer *ptr_fpga, void *ptr_cpu, long int size) {
#ifdef FPGA_DEBUG
    printf("    (copy memory from fpga: size %d, ptr_cpu %p)\n", size, ptr_cpu);
#endif
    cl_int err;
    cl::Event event;
    OCL_CHECK(err, err= q.enqueueReadBuffer(*ptr_fpga, CL_TRUE, 0, size, ptr_cpu, nullptr, &event));
    q.finish();
}



// ---------------------------------------------------
// Support functions

// -----------------------------------------------------------------
// all
//
void fpga_cpuemu_transpose(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_transpose(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_transpose(Tensor * A, Tensor * B) {
    _profile_fpga(_FPGA_TRANSPOSE, 0);
#ifndef K_ENABLED_TRANSPOSE
    fpga_cpuemu_transpose(A, B);
#else
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_transpose.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_transpose.setArg(1, *(B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_transpose.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_transpose, NULL, &event));
    q.finish();

#endif
    _profile_fpga(_FPGA_TRANSPOSE, 1);
}

// -----------------------------------------------------------------
// copy
//
void fpga_cpuemu_copy(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_copy(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_copy(Tensor * A, Tensor * B){
    _profile_fpga(_FPGA_COPY, 0);
#ifndef K_ENABLED_COPY
    fpga_cpuemu_copy(A, B);
#else
    int Asize = A->size * sizeof(float);
    if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
    fpga_copy_from_fpga(A, A->ptr);
    fpga_copy_to_fpga(A->ptr, B);
#endif
    _profile_fpga(_FPGA_COPY, 1);
}

// -----------------------------------------------------------------
// fill_
//
void fpga_cpuemu_fill_(Tensor *A, float v) {
  cpu_fill_(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_fill_(Tensor *A, float v){
    _profile_fpga(_FPGA_FILL_, 0);
#ifndef K_ENABLED_FILL_
    fpga_cpuemu_fill_(A, v);
#else
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_fill_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_fill_.setArg(1, v));
    OCL_CHECK(err, err = kernel_fill_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_fill_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_FILL_, 1);
}

// -----------------------------------------------------------------
// fill
//
void fpga_cpuemu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_fill(A, aini, aend, B, bini, bend, inc);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc){
    _profile_fpga(_FPGA_FILL, 0);
#ifndef K_ENABLED_FILL
    fpga_cpuemu_fill(A, aini, aend, B, bini, bend, inc);
#else
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_fill.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_fill.setArg(1, (int)aini));
    OCL_CHECK(err, err = kernel_fill.setArg(2, (int)aend));
    OCL_CHECK(err, err = kernel_fill.setArg(3, *(B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_fill.setArg(4, (int)bini));
    OCL_CHECK(err, err = kernel_fill.setArg(5, (int)bend));
    OCL_CHECK(err, err = kernel_fill.setArg(6, (int)inc));
    OCL_CHECK(err, err = kernel_fill.setArg(7, (int)A->ndim));
    OCL_CHECK(err, err = kernel_fill.setArg(8, (long int)A->size));
    OCL_CHECK(err, err = kernel_fill.setArg(9, (int)A->shape[0]));
    OCL_CHECK(err, err = kernel_fill.setArg(10, (int)B->size));
    OCL_CHECK(err, err = kernel_fill.setArg(11, (int)B->shape[0]));

    OCL_CHECK(err, err = q.enqueueTask(kernel_fill, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_FILL, 1);
}

// -----------------------------------------------------------------
// select
//
void fpga_cpuemu_select(Tensor *A, Tensor *B, SelDescriptor *sd) {
  int ADDRsize = B->size * sizeof(int);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_addresses_from_fpga(sd, ADDRsize, sd->cpu_addresses);
  cpu_select(A, B, sd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile_fpga(_FPGA_SELECT, 0);
#ifndef K_ENABLED_SELECT
    fpga_cpuemu_select(A, B, sd);
#else
    printf("fpga_select not implemented yet\n"); exit(1);
    // cl_int err;
    // cl::Event event;
    //
    // OCL_CHECK(err, err = kernel_select.setArg(0, *(A->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_select.setArg(1, *(B->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_select.setArg(2, ((int)sd->fpga_addresses))); //TOCHECK
    // OCL_CHECK(err, err = kernel_select.setArg(3, (long int)A->size));
    //
    // OCL_CHECK(err, err = q.enqueueTask(kernel_select, NULL, &event));
    // q.finish();
#endif
    _profile_fpga(_FPGA_SELECT, 1);
}

// -----------------------------------------------------------------
// select_back
//
void fpga_cpuemu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd) {
  int ADDRsize = B->size * sizeof(int);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_addresses_from_fpga(sd, ADDRsize, sd->cpu_addresses);
  cpu_select_back(A, B, sd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile_fpga(_FPGA_SELECT_BACK, 0);
#ifndef K_ENABLED_SELECT_BACK
    fpga_cpuemu_select_back(A, B, sd);
#else
    printf("fpga_select_back not implemented yet\n"); exit(1);
    // cl_int err;
    // cl::Event event;
    //
    // OCL_CHECK(err, err = kernel_select_back.setArg(0, *(A->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_select_back.setArg(1, *(B->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_select_back.setArg(2, ((int)sd->fpga_addresses))); //TOCHECK
    // OCL_CHECK(err, err = kernel_select_back.setArg(3, (long int)A->size));
    //
    // OCL_CHECK(err, err = q.enqueueTask(kernel_select_back, NULL, &event));
    // q.finish();
#endif
    _profile_fpga(_FPGA_SELECT_BACK, 1);
}

// -----------------------------------------------------------------
// set_select
//
void fpga_cpuemu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd) {
  int ADDRsize = B->size * sizeof(int);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_addresses_from_fpga(sd, ADDRsize, sd->cpu_addresses);
  cpu_set_select(A, B, sd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile_fpga(_FPGA_SET_SELECT, 0);
#ifndef K_ENABLED_SET_SELECT
    fpga_cpuemu_set_select(A, B, sd);
#else
    printf("fpga_set_select not implemented yet\n"); exit(1);
    // cl_int err;
    // cl::Event event;
    //
    // OCL_CHECK(err, err = kernel_set_select.setArg(0, *(A->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_set_select.setArg(1, *(B->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_set_select.setArg(2, ((int)sd->fpga_addresses))); //TOCHECK
    // OCL_CHECK(err, err = kernel_set_select.setArg(3, (long int)A->size));
    //
    // OCL_CHECK(err, err = q.enqueueTask(kernel_set_select, NULL, &event));
    // q.finish();
#endif
    _profile_fpga(_FPGA_SET_SELECT, 1);
}

// -----------------------------------------------------------------
// set_select_back
//
void fpga_cpuemu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd) {
  int ADDRsize = B->size * sizeof(int);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_addresses_from_fpga(sd, ADDRsize, sd->cpu_addresses);
  cpu_set_select_back(A, B, sd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    _profile_fpga(_FPGA_SET_SELECT_BACK, 0);
#ifndef K_ENABLED_SET_SELECT_BACK
    fpga_cpuemu_set_select_back(A, B, sd);
#else
    printf("fpga_set_select_back not implemented yet\n"); exit(1);
    // cl_int err;
    // cl::Event event;
    //
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(0, *(A->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(1, *(B->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(2, ((int)sd->fpga_addresses))); //TOCHECK
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(3, (long int)A->size));
    //
    // OCL_CHECK(err, err = q.enqueueTask(kernel_set_select_back, NULL, &event));
    // q.finish();
#endif
    _profile_fpga(_FPGA_SET_SELECT_BACK, 1);
}

// -----------------------------------------------------------------
// select2
//
void fpga_cpuemu_select(Tensor * A, Tensor * B, vector<int> sind, int ini, int end,bool mask_zeros) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_select(A, B, sind, ini, end, mask_zeros);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_select(Tensor * A, Tensor * B, vector<int> sind, int ini, int end,bool mask_zeros){
    _profile_fpga(_FPGA_SELECT2, 0);
#ifndef K_ENABLED_SELECT
    fpga_cpuemu_select(A, B, sind, ini, end, mask_zeros);
#else
    printf("fpga_select not implemented yet\n"); exit(1);
    // cl_int err;
    // cl::Event event;
    //
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(0, *(A->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(1, *(B->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(2, (sind))); //TOCHECK
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(3, (int)ini));
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(4, (int)end));
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(5, (bool)mask_zeros));
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(6, (long int)A->size));
    // OCL_CHECK(err, err = kernel_set_select_back.setArg(7, (int)A->shape[0]));
    //
    // OCL_CHECK(err, err = q.enqueueTask(kernel_set_select_back, NULL, &event));
    // q.finish();
#endif
    _profile_fpga(_FPGA_SELECT2, 1);
}

// -----------------------------------------------------------------
// deselect
//
void fpga_cpuemu_deselect(Tensor * A, Tensor * B, vector<int> sind, int ini, int end, int inc, bool mask_zeros) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_deselect(A, B, sind, ini, end, inc, mask_zeros);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_deselect(Tensor * A, Tensor * B, vector<int> sind, int ini, int end,int inc,bool mask_zeros){
    _profile_fpga(_FPGA_DESELECT, 0);
#ifndef K_ENABLED_DESELECT
    fpga_cpuemu_deselect(A, B, sind, ini, end, inc, mask_zeros);
#else
    printf("fpga_deselect not implemented yet\n"); exit(1);
    // cl_int err;
    // cl::Event event;
    //
    // OCL_CHECK(err, err = kernel_deselect.setArg(0, *(A->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_deselect.setArg(1, *(B->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_deselect.setArg(2, (sind))); //TOCHECK
    // OCL_CHECK(err, err = kernel_deselect.setArg(3, (int)ini));
    // OCL_CHECK(err, err = kernel_deselect.setArg(4, (int)end));
    // OCL_CHECK(err, err = kernel_deselect.setArg(5, (int)inc));
    // OCL_CHECK(err, err = kernel_deselect.setArg(6, (bool)mask_zeros));
    // OCL_CHECK(err, err = kernel_deselect.setArg(7, (long int)A->size));
    // OCL_CHECK(err, err = kernel_deselect.setArg(8, (int)A->shape[0]));
    //
    // OCL_CHECK(err, err = q.enqueueTask(kernel_deselect, NULL, &event));
    // q.finish();
#endif
    _profile_fpga(_FPGA_DESELECT, 1);
}

// -----------------------------------------------------------------
// concat
//
void fpga_cpuemu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative) {
  for (unsigned int i = 0; i < t.size(); i++) fpga_copy_from_fpga(t[i], t[i]->ptr);
  cpu_concat(A, t, axis, derivative);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative){
    _profile_fpga(_FPGA_CONCAT, 0);
#ifndef K_ENABLED_CONCAT
    fpga_cpuemu_concat(A, t, axis, derivative);
#else
    printf("fpga_concat not implemented yet\n"); exit(1);
#endif
    _profile_fpga(_FPGA_CONCAT, 1);
}

// -----------------------------------------------------------------
// sort
//

void fpga_cpuemu_sort(Tensor *A, Tensor *B, bool descending, bool stable) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sort(A, B, descending, stable);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_sort(Tensor *A, Tensor *B, bool descending, bool stable) {
  _profile_fpga(_FPGA_SORT, 0);
#ifndef K_ENABLED_SORT
  fpga_cpuemu_sort(A, B, descending, stable);
#else
  printf("fpga_sort not implemented yet\n");
  exit(1);
#endif
  _profile_fpga(_FPGA_SORT, 1);
}

// ----------------------------------------------------------------
// argsort
//

void fpga_cpuemu_argsort(Tensor *A, Tensor *B, bool descending, bool stable) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_argsort(A, B, descending, stable);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_argsort(Tensor *A, Tensor *B, bool descending, bool stable) {
  _profile_fpga(_FPGA_ARGSORT, 0);
#ifndef K_ENABLED_ARGSORT
  fpga_cpuemu_argsort(A, B, descending, stable);
#else
  printf("fpga_argsort not implemented yet\n");
  exit(1);
#endif
  _profile_fpga(_FPGA_ARGSORT, 1);
}

#endif
