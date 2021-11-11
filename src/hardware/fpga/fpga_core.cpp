/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

// Headers -------------------------------------------------------------------------------------------------------------------------------
#include "eddl/hardware/fpga/xcl2.hpp"      // OpenCL header
#include <vector>                           // Vectors
#include <math.h>                           // Math functions
#include <float.h>                          // Float operations
#include "eddl/tensor/tensor.h"             // EDDL Tensors
#include "eddl/descriptors/descriptors.h"   // EDDL Descriptors
#include "eddl/hardware/fpga/fpga_hw.h"     // FPGA enables of kernels
#include <sys/time.h>                       // Time (for stats)
#include "eddl/hardware/cpu/cpu_tensor.h"   // CPU related function headers (cpu_transpose, cpu_copy, ...)
#include <ap_fixed.h>                       // Aproximated precision fixed point support
#include <ap_int.h>                         // Aproximated precision integer support
#include "eddl/profiling.h"                 // Profiling

// Macros ---------------------------------------------------------------------------------------------------------------------------------
PROFILING_ENABLE_EXTERN(Precision_Conversion);
PROFILING_ENABLE_EXTERN(FPGA_READ);
PROFILING_ENABLE_EXTERN(FPGA_WRITE);


// Defines --------------------------------------------------------------------------------------------------------------------------------
#define MAX_BUFFER_POOL 10000             // All FPGA tensors are managed in a pool, this define sets the pool size (static)


// Global variables -----------------------------------------------------------------------------------------------------------------------
int next_fpga_tensor_id = 1;              // Each FPGA tensor is identified with a unique ID, this is the counter of the next available ID
int num_tensors_created = 0;              // The tensors created in FPGA are managed in a queue, this is the counter of created tensors

cl::Buffer *fpga_ptr_buffer_pool[MAX_BUFFER_POOL];    // pool of tensors created in the FPGA
long fpga_size_buffer_pool[MAX_BUFFER_POOL];          // size of each buffer in the pool
int fpga_inuse_buffer_pool[MAX_BUFFER_POOL];          // flag for each slot in the pool indicating whether is in use or not (can be reused)
int fpga_free_buffer_pool[MAX_BUFFER_POOL];           // flag for each slot in the pool indicating whether is empty or has valid information
int fpga_num_buffer_pool_slots;                       // number of allocated slots in the pool

cl::Context               *context;                   // OpenCL context
std::vector<cl:: Device>   devices;                   // List of OpenCL devices
cl::Device                 device;                    // FPGA device
cl::CommandQueue          *q;                         // Command queue
cl::CommandQueue           com;                       // Command queue
cl::Program               *program;                   // Program

vector<cl::Event> kernel_events(MAX_KERNELS);         // Kernel events (completion)

// ---------------------------------------------------------------------------------------------------------------------------------------
// List of all kernels that can be instantiated in the FPGA
//
// activation kernels (22)
cl::Kernel kernel_relu,   kernel_d_relu,  kernel_thresholded_relu,    kernel_d_thresholded_relu, kernel_leaky_relu,     kernel_d_leaky_relu;
cl::Kernel kernel_elu,    kernel_d_elu,   kernel_softplus,            kernel_d_softplus,         kernel_softsign,       kernel_d_softsign;
cl::Kernel kernel_linear, kernel_d_linear,kernel_sigmoid,             kernel_d_sigmoid,          kernel_hard_sigmoid,   kernel_d_hard_sigmoid;
cl::Kernel kernel_exp,    kernel_d_exp,   kernel_tanh, kernel_d_tanh, kernel_softmax,            kernel_d_softmax;
//
// bn kernels (4)
cl::Kernel kernel_permute_channels_last, kernel_permute_channels_first;
cl::Kernel kernel_permute_batch_last,    kernel_permute_batch_first;
//
// comparison kernels (20)
cl::Kernel kernel_all,         kernel_any,        kernel_isfinite,    kernel_isinf;
cl::Kernel kernel_isnan,       kernel_isneginf,   kernel_isposinf,    kernel_equal2;
cl::Kernel kernel_logical_and, kernel_logical_or, kernel_logical_not, kernel_logical_xor;
cl::Kernel kernel_allclose,    kernel_isclose,    kernel_greater,     kernel_greater_equal;
cl::Kernel kernel_less,        kernel_less_equal, kernel_equal,       kernel_not_equal;
cl::Kernel kernel_greater_vector, kernel_greater_equal_vector, kernel_less_vector;
cl::Kernel kernel_less_equal_vector, kernel_equal_vector, kernel_not_equal_vector;
//
// core kernels (11)
cl::Kernel kernel_transpose,   kernel_copy,        kernel_fill_,      kernel_fill;
cl::Kernel kernel_select,      kernel_select_back, kernel_set_select, kernel_set_select_back;
cl::Kernel kernel_set_select2, kernel_deselect,    kernel_concat;
cl::Kernel kernel_select_nn,   kernel_select_back_nn, kernel_set_select_nn, kernel_set_select_back_nn;
//
// conv kernels (3)
cl::Kernel kernel_im2col;
cl::Kernel kernel_conv2D[16];
//
// create kernels (3)
cl::Kernel kernel_range, kernel_eye, kernel_diag;
//
// da kernels (6)
cl::Kernel kernel_single_shift, kernel_single_rotate, kernel_single_scale;
cl::Kernel kernel_single_flip,  kernel_single_crop;
cl::Kernel kernel_crop_scale_random;
//
// generator kernels (4)
cl::Kernel kernel_rand_uniform, kernel_signed_uniform, kernel_rand_binary, kernel_rand_normal;
//
// losses kernels (1)
cl::Kernel kernel_cent;
//
// metrics kernels (22)
cl::Kernel kernel_accuracy, kernel_bin_accuracy;
//
// pool kernels (4)
cl::Kernel kernel_mpool2D, kernel_mpool2D_back, kernel_avgpool2D, kernel_avgpool2D_back;
//
// reduction kernels (5)
cl::Kernel kernel_reduce, kernel_reduce_op, kernel_reduce_sum2D, kernel_reduction, kernel_reduction_back;
//
// tensor_nn kernels (2)
cl::Kernel kernel_repeat_nn, kernel_d_repeat_nn;
//
// math kernels (46)
cl::Kernel kernel_abs,       kernel_acos,   kernel_add,      kernel_asin,       kernel_atan,          kernel_ceil,          kernel_clamp;
cl::Kernel kernel_cos,       kernel_cosh,   kernel_mod,      kernel_mult,       kernel_trunc,         kernel_sum_abs;
cl::Kernel kernel_floor,     kernel_inv,    kernel_log,      kernel_log2,       kernel_log10,         kernel_logn;
cl::Kernel kernel_normalize, kernel_pow,    kernel_powb,     kernel_reciprocal, kernel_remainder,     kernel_round,         kernel_rsqrt;
cl::Kernel kernel_sign,      kernel_sin,    kernel_sinh,     kernel_sqr,        kernel_sqrt,          kernel_tan;
cl::Kernel kernel_inc,       kernel_el_div, kernel_el_mult,  kernel_sign2,      kernel_sum2D_rowwise, kernel_sum2D_colwise;
cl::Kernel kernel_max,       kernel_min,    kernel_sum,      kernel_mult2d;

// -------------------------------------------------------------------------------------------------------------------------------------------
// conv2d kernel related global variables
#ifdef K_ENABLED_CONV2D
int k_conv2d_cpi;             // number of CPI channels (parallel channels read)
int k_conv2d_cpo;             // number of CPO channels (parallel channels written)
int k_conv2d_num_kernels;     // number of kernels present in the FPGA
int k_conv2d_max_rows;        // maximum number of rows that the kernel can handle
int k_conv2d_max_ho;          // maximum number of output rows the kernel can handle
int k_conv2d_max_wo;          // maximum number of output cols the kernel can handle
#endif

// -------------------------------------------------------------------------------------------------------------------------------------------
// Global variables for profiling
// Each kernel can be profiled (obtained the number of instances executed and the accumulated execution time)
//
int num_instances_fpga[_NUM_FPGA_FUNCS];            // number of instances a kernel (function) has been called
struct timeval time_ini_fpga[_NUM_FPGA_FUNCS];      // initial time of an instance for a kernel (function). Temporary variable
unsigned long long acc_time_fpga[_NUM_FPGA_FUNCS];  // accumulated ime of a kernel (function)

// profiling of FPGA resources being used
float mb_memory_needed_fpga;                        // Megabytes of memory needed for tensors in the FPGA


// OpenCL-related support functions ----------------------------------------------------------------------------------------------------------
//

// set_callback(). Sets the callback for a particular event in OpenCL
void set_callback(cl::Event event, const char *queue_name) {
  cl_int err;
  OCL_CHECK(err, err = event.setCallback(CL_COMPLETE, event_cb, (void *)queue_name));
}

// event_cb(). An event callback function that prints the operations performed by the OpenCL runtime
void event_cb(cl_event event1, cl_int cmd_status, void *data) {
  #ifdef FPGA_DEBUG
  cl_int err;
  cl_command_type command;
  cl::Event event(event1, true);
  OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_TYPE, &command));
  cl_int status;
  OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status));

  const char *command_str;
  const char *status_str;
  switch (command) {
    case CL_COMMAND_READ_BUFFER:          command_str = "buffer read";    break;
    case CL_COMMAND_WRITE_BUFFER:         command_str = "buffer write";   break;
    case CL_COMMAND_NDRANGE_KERNEL:       command_str = "kernel";         break;
    case CL_COMMAND_MAP_BUFFER:           command_str = "kernel";         break;
    case CL_COMMAND_COPY_BUFFER:          command_str = "kernel";         break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:  command_str = "buffer migrate"; break;
    default:                              command_str = "unknown";
  }
  switch (status) {
    case CL_QUEUED:    status_str = "Queued";    break;
    case CL_SUBMITTED: status_str = "Submitted"; break;
    case CL_RUNNING:   status_str = "Executing"; break;
    case CL_COMPLETE:  status_str = "Completed"; break;
  }
  printf("[%s]: %s %s\n", reinterpret_cast<char *>(data), status_str, command_str);
  fflush(stdout);
  #endif
}

// _profile_fpga_funcname(). profiling function
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
      case _FPGA_CONV2D_RELU             : strcpy(name, "conv2d_relu"); break;
      case _FPGA_CONV2D_STM             : strcpy(name, "conv2d_stm"); break;
      case _FPGA_CONV2D_MAXPOOL         : strcpy(name, "conv2d_maxpool"); break;
      case _FPGA_CONV2D_RELU_MAXPOOL    : strcpy(name, "conv2d_relu_maxpool"); break;
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
      case _FPGA_SUM_2                  : strcpy(name, "sum_2"); break;
      case _FPGA_TRANSFORM              : strcpy(name, "transform"); break;
      default                          : strcpy(name, "?????"); break;
  }
}

// _profile_fpga(). Function to profile a kernel (function)
void _profile_fpga(int f_id, int end) {
  #ifdef FPGA_DEBUG
  char func_name[50];
  _profile_fpga_funcname(f_id, func_name);
  if (!end) printf("FPGA_DEBUG: Function %s starts\n", func_name);
  if (end)  printf("FPGA_DEBUG: Function %s ends\n", func_name);
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

// profile_fpga_tensor(). Function to profile a tensor.
// It provides tensor information through the console
void _profile_fpga_tensor(Tensor *T) {
  #ifdef FPGA_DEBUG
  // We read the tensor from FPGA first
  fpga_copy_from_fpga(T, T->ptr);
  // Now we calculate statistics (min, max, avg) from the tensor
  float min = FLT_MAX;
  float max = -FLT_MAX;
  float sum = 0.f;
  float avg;
  for (int i=0; i<T->size; i++) {
    if (T->ptr[i] > max) max = T->ptr[i];
    if (T->ptr[i] < min) min = T->ptr[i];
    sum += T->ptr[i];
  }
  avg = sum / (float)T->size;

  // Now, we print the information (related tensor information and statistics of the tensor)
  printf("FPGA_DEBUG: Tensor: %3d ", T->fpga_tensor_id);
  printf(" size %10d ", T->size);
  //printf(" size_fpga %10d ", T->fpga_size);
  printf(" shape ");
  if (T->ndim==1) printf("%d", T->shape[0]);
  if (T->ndim==2) printf("%dx%d", T->shape[0], T->shape[1]);
  if (T->ndim==3) printf("%dx%dx%d", T->shape[0], T->shape[1], T->shape[2]);
  if (T->ndim==3) printf("%dx%dx%dx%d", T->shape[0], T->shape[1], T->shape[2], T->shape[3]);
  printf(" Min %8.4f Max %8.4f Avg %8.4f\n", min, max, avg);
  #endif
}

// _profile_fpga_tensor_print(). Prints some values of the tensor
void _profile_fpga_tensor_print(Tensor *T) {
  #ifdef FPGA_DEBUG_VERBOSE
  // We read the tensor from FPGA
  printf("tensor print:\n");
  fpga_copy_from_fpga(T, T->ptr);
  int d1_max = 2;
  int d2_max = 4;
  int d3_max = 4;
  if (T->ndim==4) {
    for (int d0=0; d0<T->shape[0]; d0++) {
    for (int d1=0; d1<d1_max; d1++) {
    for (int d2=0; d2<d2_max; d2++) {
    for (int d3=0; d3<d3_max; d3++) {
    
    //for (int d0=0; d0<T->shape[0]; d0++) {
    //for (int d1=0; d1<T->shape[1]; d1++) {
    //for (int d2=0; d2<T->shape[2]; d2++) {
    //for (int d3=0; d3<T->shape[3]; d3++) {
      int a = (d0 * T->shape[1] * T->shape[2] * T->shape[3]) + (d1 * T->shape[2] * T->shape[3]) + (d2 * T->shape[3]) + d3;
      printf("%f ", T->ptr[a]);
      
    }
    //printf("\n");
    }
    //printf("\n\n");
    }
    //printf("\n\n\n");
    }
  }  else if(T->ndim==2) {
       for (int d0=0; d0<d1_max; d0++) {
       for (int d1=0; d1<d2_max; d1++) {
       //for (int d0=0; d0<T->shape[0]; d0++) {
       //for (int d1=0; d1<T->shape[1]; d1++) {
         int a = (d0 * T->shape[1]) + d1;
         printf("%f ", T->ptr[a]);
       }
       printf("\n\n");
    }

  } else if(T->ndim==1) {
    for (int d0=0; d0<T->shape[0]; d0++) {
      printf("%f ", T->ptr[0]);
    }
    printf("\n\n");
    }
  printf("\n");
  #endif
}

void _debug_fpga_funcs(const char *str) {
  #ifdef FPGA_DEBUG_FUNCS 
  printf("%s\n", str);
  #endif
}

// _show_profile_fpga(). Shows all the profile collected so far.
void _show_profile_fpga() {
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
}

// _profile_fpga_add_tensor(). Adds a tensor profile information
void _profile_fpga_add_tensor(int size) {
  mb_memory_needed_fpga += (float)size / 1024.0 / 1024.0;
  num_tensors_created++;
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (accumulated tensor memory %f MB)\n", mb_memory_needed_fpga);
  #endif
}

// _profile_fpga_remove_tensor(). Removes tensor profile related information
void _profile_fpga_remove_tensor(int size) {
  mb_memory_needed_fpga -= (float)size / 1024.0 / 1024.0;
  num_tensors_created--;
}

// -----------------------------------------------------------------------------------------------------------------------------------
// FPGA initialization and finalization functions

// fpga_init()
// Initialices the device, sets up the kernels, prepares everything related to the FPGA device and support infrastructure
// This function must be called only once and at the begining of operations with the FPGA
void fpga_init(){

  if (context!=NULL) {
    #ifdef FPGA_DEBUG
    printf("fpga_init function skipped, previous initialization done\n");
    #endif
    return;
  }

  #ifdef FPGA_DEBUG
  printf("initializing FPGA\n");
  #endif

  cl_int      err;

  // depending on the kernel version we load one binary file or another
  std::string binaryFile;

  #ifdef K_ENABLED_CONV2D
  // We need to instantiate the proper number of kernels, we also take the specifities of the kernels
  int kernel_version = K_VERSION_CONV2D;
  int kernel_subversion = K_SUBVERSION_CONV2D;
  printf("kernel version %d.%d\n", kernel_version, kernel_subversion);
  switch (kernel_version) {
    case 1: switch (kernel_subversion) {
              case 0: k_conv2d_cpi = 4; k_conv2d_cpo = 4; k_conv2d_num_kernels = 1; k_conv2d_max_rows = 256; binaryFile = "conv2D_4x4_fp32_relu_1kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
              case 1: k_conv2d_cpi = 4; k_conv2d_cpo = 4; k_conv2d_num_kernels = 1; k_conv2d_max_rows = 256; binaryFile = "conv2D_4x4_apf8_relu_1kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
              case 2: k_conv2d_cpi = 4; k_conv2d_cpo = 4; k_conv2d_num_kernels = 1; k_conv2d_max_rows = 256; binaryFile = "conv2D_4x4_api8_relu_1kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
              case 3: k_conv2d_cpi = 8; k_conv2d_cpo = 8; k_conv2d_num_kernels = 1; k_conv2d_max_rows = 256; binaryFile = "conv2D_8x8_apf8_relu_1kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
              case 4: k_conv2d_cpi = 8; k_conv2d_cpo = 8; k_conv2d_num_kernels = 1; k_conv2d_max_rows = 256; binaryFile = "conv2D_8x8_api8_relu_1kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
              case 5: k_conv2d_cpi = 4; k_conv2d_cpo = 4; k_conv2d_num_kernels = 2; k_conv2d_max_rows = 256; binaryFile = "conv2D_4x4_fp32_relu_2kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
              default: printf("Error, unrecognized conv2d kernel subversion\n"); exit(1); break;
            }
	    break;
    // Version 2: Kernels with GIHWCPI format 
    case 2: switch (kernel_subversion) {
	      case 0: k_conv2d_cpi = 4; k_conv2d_cpo = 4; k_conv2d_num_kernels = 1; k_conv2d_max_rows = 256; binaryFile = "conv2D_v2.0_4x4_fp32_1kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 1024; break;
        case 1: k_conv2d_cpi = 4; k_conv2d_cpo = 4; k_conv2d_num_kernels = 1; k_conv2d_max_rows = 256; binaryFile = "conv2D_v2.0_4x4_fp32_stm_bn_1kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
        case 2: k_conv2d_cpi = 4; k_conv2d_cpo = 4; k_conv2d_num_kernels = 2; k_conv2d_max_rows = 256; binaryFile = "conv2D_v2.0_4x4_fp32_stm_bn_2kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 1024; break;
        case 3: k_conv2d_cpi = 8; k_conv2d_cpo = 8; k_conv2d_num_kernels = 1; k_conv2d_max_rows = 256; binaryFile = "conv2D_v2.0_8x8_fp32_stm_bn_1kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
        case 4: k_conv2d_cpi = 8; k_conv2d_cpo = 8; k_conv2d_num_kernels = 2; k_conv2d_max_rows = 256; binaryFile = "conv2D_v2.0_8x8_fp32_stm_bn_2kernel.xclbin"; k_conv2d_max_ho = 256; k_conv2d_max_wo = 256; break;
	      default: printf("Error, unrecognized conv2d kernel subversion\n"); exit(1); break;
	    }
	    break;
    default: printf("Error, unrecognized conv2d kernel version\n"); exit(1); break;
  };
  #else
  binaryFile = "eddl.xclbin";
  #endif

  unsigned    fileBufSize;

  #ifdef FPGA_DEBUG
  std::cout << "Creating Context..." << std::endl;
  #endif
  
  devices = xcl::get_xil_devices();
  device = devices[0];

  OCL_CHECK(err, context = new cl::Context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, q = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    
  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins;

  bins = cl::Program::Binaries{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);

  OCL_CHECK(err, program = new cl::Program(*context, devices, bins, NULL, &err));

  #ifdef FPGA_DEBUG
  std::cout << "Device " << device_name.c_str() << ": program successful!" << std::endl;
  #endif

  // Now, we instatiate every possible kernel (enabled by the proper define)
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
  for (int k=0; k<k_conv2d_num_kernels; k++) {
    char dummy[50];
    sprintf(dummy, "k_conv2D:{k_conv2D_%d}", k+1);
    OCL_CHECK(err, kernel_conv2D[k] = cl::Kernel(*program, dummy, &err));
    std::cout << "Kernel sucessfully created" << std::endl ;
  }
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

  #ifdef K_ENABLED_ABS_
  OCL_CHECK(err, kernel_abs = cl::Kernel(program,"k_abs", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_ACOS_
  OCL_CHECK(err, kernel_acos = cl::Kernel(program,"k_acos", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_ADD_
  OCL_CHECK(err, kernel_add = cl::Kernel(program,"k_add", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_ASIN_
  OCL_CHECK(err, kernel_asin = cl::Kernel(program,"k_asin", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_ATAN_
  OCL_CHECK(err, kernel_atan = cl::Kernel(program,"k_atan", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_CEIL_
  OCL_CHECK(err, kernel_ceil = cl::Kernel(program,"k_ceil", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_CLAMP_
  OCL_CHECK(err, kernel_clamp = cl::Kernel(program,"k_clamp", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_K_COS_
  OCL_CHECK(err, kernel_cos = cl::Kernel(program,"k_cos", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_COSH_
  OCL_CHECK(err, kernel_cosh = cl::Kernel(program,"k_cosh", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_FLOOR_
  OCL_CHECK(err, kernel_floor = cl::Kernel(program,"k_floor", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_INV_
  OCL_CHECK(err, kernel_inv = cl::Kernel(program,"k_inv", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_LOG_
  OCL_CHECK(err, kernel_log = cl::Kernel(program,"k_log", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_LOG2_
  OCL_CHECK(err, kernel_log2 = cl::Kernel(program,"k_log2", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_LOG10_
  OCL_CHECK(err, kernel_log10 = cl::Kernel(program,"k_log10", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_LOGN_
  OCL_CHECK(err, kernel_logn = cl::Kernel(program,"k_logn", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_MOD_
  OCL_CHECK(err, kernel_mod = cl::Kernel(program,"k_mod", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_MULT_
  OCL_CHECK(err, kernel_mult = cl::Kernel(program,"k_mult", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_NORMALIZE_
  OCL_CHECK(err, kernel_normalize = cl::Kernel(program,"k_normalize", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_POW_
  OCL_CHECK(err, kernel_pow = cl::Kernel(program,"k_pow", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_POWB_
  OCL_CHECK(err, kernel_powb = cl::Kernel(program,"k_powb", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_RECIPROCAL_
  OCL_CHECK(err, kernel_reciprocal = cl::Kernel(program,"k_reciprocal", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_REMAINDER_
  OCL_CHECK(err, kernel_remainder = cl::Kernel(program,"k_remainder", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_ROUND_
  OCL_CHECK(err, kernel_round = cl::Kernel(program,"k_round", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_RSQRT_
  OCL_CHECK(err, kernel_rsqrt = cl::Kernel(program,"k_rsqrt", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_SIGN_
  OCL_CHECK(err, kernel_sign = cl::Kernel(program,"k_sign", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_SIN_
  OCL_CHECK(err, kernel_sin = cl::Kernel(program,"k_sin", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_SINH_
  OCL_CHECK(err, kernel_sinh = cl::Kernel(program,"k_sinh", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_SQR_
  OCL_CHECK(err, kernel_sqr = cl::Kernel(program,"k_sqr", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_SQRT_
  OCL_CHECK(err, kernel_sqrt = cl::Kernel(program,"k_sqrt", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_TAN_
  OCL_CHECK(err, kernel_tan = cl::Kernel(program,"k_tan", &err));
  if (err != CL_SUCCESS) printf("Error creating kernel\n");
  #endif

  #ifdef K_ENABLED_TRUNC_
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

  // Initializing buffer pool
  for (int e=0; e<MAX_BUFFER_POOL; e++) {
    fpga_ptr_buffer_pool[e] = (cl::Buffer *)nullptr;
    fpga_size_buffer_pool[e] = 0;
    fpga_inuse_buffer_pool[e] = 0;
    fpga_free_buffer_pool[e] = 1;
  }
  fpga_num_buffer_pool_slots = 0;
  
  #ifdef FPGA_DEBUG
  printf("end of fpga_init\n");
  #endif
}

void close_fpga() {}


// ------------------------------------------------------------------------------------------------------------
// Tensor creation and delete functions
//
cl::Buffer *fpga_create_tensor(int device, int size) {
  cl::Buffer *buffer;
  cl_int err;

  _profile_fpga_add_tensor(size*sizeof(fpga_data_type));

  // search an available slot
  int e;
  for (e=0; e<fpga_num_buffer_pool_slots; e++) {
    if (!fpga_inuse_buffer_pool[e] && !fpga_free_buffer_pool[e] & (fpga_size_buffer_pool[e] == size)) break;
  }
  if (e!=fpga_num_buffer_pool_slots) {
    #ifdef FPGA_DEBUG_VERBOSE
    printf("    reasigning buffer pool entry\n");
    #endif
  
    fpga_inuse_buffer_pool[e] = 1;

    #ifdef FPGA_DEBUG
    printf("FPGA_DEBUG: Allocation of tensor. Size %12d elements. Address %p. Pool entry %4d\n", size, fpga_ptr_buffer_pool[e], e);
    #endif

    return fpga_ptr_buffer_pool[e];
  }
  
  // create a new buffer pool
  if (fpga_num_buffer_pool_slots == MAX_BUFFER_POOL) {
    printf("Error, too many buffer pools\n");
    exit(1);
  }

  // buffer pool slot creation
  #ifdef FPGA_DEBUG_VERBOSE
  printf("Creating new buffer pool entry\n");
  #endif

  e = fpga_num_buffer_pool_slots;
  cl_mem_ext_ptr_t ext = {0};
  //ext.banks = XCL_MEM_DDR_BANK0;
  ext.flags  = 0 | XCL_MEM_TOPOLOGY;
  OCL_CHECK(err,buffer = new cl::Buffer(*context,CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, size*sizeof(fpga_data_type), &ext, &err));
  fpga_ptr_buffer_pool[e] = buffer;
  fpga_size_buffer_pool[e] = size;
  fpga_inuse_buffer_pool[e] = 1;
  fpga_free_buffer_pool[e] = 0;
  fpga_num_buffer_pool_slots++;

  #ifdef FPGA_DEBUG
  printf("FPGA_DEBUG: Allocation of tensor. Size %12d elements. Address %p. Pool entry %4d\n", size, fpga_ptr_buffer_pool[e], e);
  #endif

  return fpga_ptr_buffer_pool[e];
}


void fpga_delete_tensor(int device, cl::Buffer *ptr, int fpga_tensor_id_p, int size) {
  #ifdef FPGA_DEBUG
  printf("    (deleting tensor in fpga, id %d)\n", fpga_tensor_id_p);
  #endif
  _profile_fpga_remove_tensor(size*sizeof(fpga_data_type));

  // we just update the buffer pool
  int e;
  for (e=0; e<fpga_num_buffer_pool_slots; e++) {
    if (fpga_inuse_buffer_pool[e] && !fpga_free_buffer_pool[e] && (fpga_size_buffer_pool[e] == size) && (fpga_ptr_buffer_pool[e] == ptr)) break;
  }
  if (e==fpga_num_buffer_pool_slots) {
    printf("Error, delete tensor function did not find the buffer in the pool\n");
    exit(1);
  }
  // we remove the buffer
  delete fpga_ptr_buffer_pool[e];
  fpga_free_buffer_pool[e] = 1;    // this entry now is not assigned to a buffer and is free
  fpga_inuse_buffer_pool[e] = 0;

  #ifdef FPGA_DEBUG
  printf("    tensor deleted\n");
  #endif
}

// ------------------------------------------------------------------------------------------------------------------------
// Copy operations
//
void fpga_copy_fpga(Tensor *A, Tensor *B) {
  #ifdef FPGA_DEBUG
  printf("    (copy fpga: tensor id %d (size %d, ptr %p) -> tensor id %d (size %d, ptr %p))\n", A->fpga_tensor_id, A->size, A->fpga_ptr, B->fpga_tensor_id, B->size, B->fpga_ptr);
  #endif
  
  cl_int err;
  cl::Event blocking_event;
  cl::Buffer *bufferA = (cl::Buffer*)A->fpga_ptr;
  cl::Buffer *bufferB = (cl::Buffer*)B->fpga_ptr;

  if (A->size > B->size) {printf("Error, copy_fpga beyond limits\n"); exit(1);}

  // TODO... enqueueCopyBuffer does not work, we need to pass through CPU!!!!!!!
  if (1==1) { //(A->size < 16) {
    float *p = (float *)malloc(A->size*sizeof(float));
    fpga_copy_from_fpga(A, p);
    fpga_copy_to_fpga(p, B);
    free(p);
  } else {
    OCL_CHECK(err, err= (*q).enqueueCopyBuffer(*bufferA, *bufferB, 0, 0, A->size*sizeof(fpga_data_type), NULL, &blocking_event));
    (*q).finish();
  }
  #ifdef FPGA_DEBUG
  printf("copy completed\n");
  #endif
}

void fpga_copy_to_fpga(float *nptr, Tensor *A, int cvt) {
  #ifdef FPGA_DEBUG
  printf("FPGA_DEBUG: Copy CPU->FPGA. Addr: %p->%p. tensor_id %4d. Size %4d\n", nptr, A->fpga_ptr, A->fpga_tensor_id, A->size);
  #endif

  #ifdef PRECISION_CONVERSION
  if (cvt) {
    _debug_fpga_funcs("Conversion (CPU->FPGA)");
    PROFILING_HEADER(Precision_Conversion);
    // We allocate a buffer to convert from floats to fpga_data_type
    fpga_data_type *cpu_buff = (fpga_data_type*)malloc(A->size*sizeof(fpga_data_type));
    for (int x=0; x<A->size; x++) cpu_buff[x] = fpga_data_type(nptr[x]);
    PROFILING_FOOTER(Precision_Conversion);
    // now we copy into the FPGA
    cl_int err;
    cl::Event blocking_event;
    cl::Buffer *buf = (cl::Buffer*)A->fpga_ptr;
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*buf, CL_TRUE, 0, A->size*sizeof(fpga_data_type), cpu_buff, nullptr, &blocking_event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_WRITE);
    free(cpu_buff);
  } else {
    // regular copy from CPU to FPGA with no precision conversion
    cl_int err;
    cl::Event blocking_event;
    cl::Buffer *buf = (cl::Buffer*)A->fpga_ptr;
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*buf, CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, nullptr, &blocking_event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_WRITE);
  }
  #else
  // regular copy from CPU to FPGA with no precision conversion
  cl_int err;
  cl::Event blocking_event;
  cl::Buffer *buf = (cl::Buffer*)A->fpga_ptr;
  PROFILING_HEADER(FPGA_WRITE);
  OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*buf, CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, nullptr, &blocking_event));
  (*q).finish();
  PROFILING_FOOTER(FPGA_WRITE);
  #endif
}

void fpga_copy_from_fpga(Tensor *A,float *nptr, int cvt) {
  #ifdef FPGA_DEBUG
  printf("FPGA_DEBUG: Copy FPGA->CPU. Addr: %p->%p. tensor_id %4d. Size %4d\n", A->fpga_ptr, nptr, A->fpga_tensor_id, A->size);
  #endif

  #ifdef PRECISION_CONVERSION
  // We read from the FPGA to a temporal buffer and then convert the precision
  if (cvt) {
    _debug_fpga_funcs("Conversion (FPGA->CPU)");
    fpga_data_type *cpu_buff = (fpga_data_type*)malloc(A->size * sizeof(fpga_data_type));
    cl_int err;
    cl::Event event;
    PROFILING_HEADER(FPGA_READ);
    OCL_CHECK(err, err= (*q).enqueueReadBuffer(*((cl::Buffer*)A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(fpga_data_type), cpu_buff, nullptr, &event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_READ);
    PROFILING_HEADER(Precision_Conversion);
    // now we perform the precision conversion
    for (int x=0; x<A->size; x++) nptr[x] = float(cpu_buff[x]);
    free(cpu_buff);
    PROFILING_FOOTER(Precision_Conversion);
  } else {
    // regular copy from FPGA to CPU with no precision conversion
    cl_int err;
    cl::Event event;
    PROFILING_HEADER(FPGA_READ);
    OCL_CHECK(err, err= (*q).enqueueReadBuffer(*((cl::Buffer*)A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, nullptr, &event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_READ);
  }
  #else
  // regular copy from FPGA to CPU with no precision conversion
  cl_int err;
  cl::Event event;
  PROFILING_HEADER(FPGA_READ);
  OCL_CHECK(err, err= (*q).enqueueReadBuffer(*((cl::Buffer*)A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, nullptr, &event));
  (*q).finish();
  PROFILING_FOOTER(FPGA_READ);
  #endif
}

void fpga_copy_addresses_from_fpga(SelDescriptor *SD, int size, int *nptr) {
  cl_int err;
  cl::Event event;
  cl::Buffer *buf = (cl::Buffer*)SD->fpga_ptr;
  PROFILING_HEADER(FPGA_READ);
  OCL_CHECK(err, err= (*q).enqueueReadBuffer(*buf, CL_TRUE, 0, size, nptr, nullptr, &event));
  (*q).finish();
  PROFILING_FOOTER(FPGA_READ);
}

void fpga_destroy_memory(cl::Buffer *fpga_ptrI) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("   destroy_memory buffer in FPGA\n");
  #endif
}

cl::Buffer *fpga_create_memory(long int size) {
  cl::Buffer *buffer;
  cl_int err;
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (creating memory in fpga size %d)\n", size);
  #endif

  OCL_CHECK(err,buffer = new cl::Buffer(*context, CL_MEM_READ_WRITE, size, NULL, &err));
  return buffer;
}

void fpga_copy_memory_to_fpga(void *ptr_cpu, cl::Buffer *ptr_fpga, long int size) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory to fpga: size %d, ptr_cpu %p)\n", size, ptr_cpu);
  #endif
  cl_int err;
  cl::Event blocking_event;
  PROFILING_HEADER(FPGA_WRITE);
  OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*ptr_fpga, CL_TRUE, 0, size, ptr_cpu, nullptr, &blocking_event));
  (*q).finish();
  PROFILING_FOOTER(FPGA_WRITE);
}

void fpga_copy_memory_from_fpga(cl::Buffer *ptr_fpga, void *ptr_cpu, long int size) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory from fpga: size %d, ptr_cpu %p)\n", size, ptr_cpu);
  #endif
  cl_int err;
  cl::Event event;
  PROFILING_HEADER(FPGA_READ);
  OCL_CHECK(err, err= (*q).enqueueReadBuffer(*ptr_fpga, CL_TRUE, 0, size, ptr_cpu, nullptr, &event));
  (*q).finish();
  PROFILING_FOOTER(FPGA_READ);
}



// ----------------------------------------------------------------------------------------------------------------------------------------
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
  #endif
  _profile_fpga(_FPGA_DESELECT, 1);
}

// -----------------------------------------------------------------
// concat
//
void fpga_cpuemu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  for (unsigned int i = 0; i < t.size(); i++) {
    int Tsize = t[i]->size * sizeof(float);
    if (t[i]->ptr == NULL) t[i]->ptr = (float *)malloc(Tsize);
    fpga_copy_from_fpga(t[i], t[i]->ptr);
  }
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

#endif

// -----------------------------------------------------------------
// transform_nn
//
void fpga_transform_nn(Tensor *A, Tensor *B, int mode) {
  _profile_fpga(_FPGA_TRANSFORM, 0);
  _profile_fpga_tensor(A);
  _debug_fpga_funcs("transform");

  #ifdef FPGA_DEBUG
 printf("fpga_transform:\n");
 printf(" A tensor: "); _profile_fpga_tensor(A);
#endif

  int CPI = k_conv2d_cpi;

  if (mode == 1) {

    // transformation from CHW to GHWC
    fpga_copy_from_fpga(A, A->ptr);
    int B_in = A->shape[0];
    int C_in = A->shape[1];
    int H_in = A->shape[2];
    int W_in = A->shape[3];
    int C_out = B->shape[1];
    // B_out, H_out and W_out assuned to be equal to B_in, H_in, W_in

    float *ptr_src = A->ptr;
    float *ptr_dst = B->ptr;

    memset(ptr_dst, 0, C_out * H_in * W_in * B_in * sizeof(float));

    for (int b=0; b<B_in; b++) {
      for (int c=0; c<C_in; c++) {
        for (int h=0; h<H_in; h++) {
          for (int w=0; w<W_in; w++) {
            int addr_src = (b * C_in * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
            int g = c / CPI;
            int cpi = c % CPI; 
            int addr_dst = (b * C_out * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
            ptr_dst[addr_dst] = ptr_src[addr_src];
          }
        }
      }
    }
    fpga_copy_to_fpga(B->ptr, B);

  } else {
    // transformation from GHWC to CHW
    fpga_copy_from_fpga(A, A->ptr);
    int B_in = A->shape[0];
    int C_in = A->shape[1];
    int H_in = A->shape[2];
    int W_in = A->shape[3];
    int C_out = B->shape[1];
    // B_out, H_out and W_out assuned to be equal to B_in, H_in, W_in

    float *ptr_src = A->ptr;
    float *ptr_dst = B->ptr;

    memset(ptr_dst, 0, C_out * H_in * W_in * B_in * sizeof(float));

    for (int b=0; b<B_in; b++) {
      for (int c=0; c<C_in; c++) {
        for (int h=0; h<H_in; h++) {
          for (int w=0; w<W_in; w++) {
            int g = c / CPI;
            int cpi = c % CPI; 
            int addr_src = (b * C_in * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
            int addr_dst = (b * C_out * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
            ptr_dst[addr_dst] = ptr_src[addr_src];
          }
        }
      }
    }
    fpga_copy_to_fpga(B->ptr, B);
  }

  _profile_fpga(_FPGA_TRANSFORM, 1);
  _profile_fpga_tensor(B);
  _debug_fpga_funcs("end transform");
#ifdef FPGA_DEBUG
  printf(" B tensor: "); _profile_fpga_tensor(B);
#endif
}


void filter_IHW_to_GIHWCPI(Tensor *A, Tensor *B) {

      float *src_ptr = A->ptr;
      float *dst_ptr = B->ptr;

      int src_I = A->shape[1];
      int src_O = A->shape[0];

      int dst_I = B->shape[1];
      int dst_O = B->shape[0];

      int KH = A->shape[2];
      int KW = A->shape[3];

      int dst_KH = B->shape[2];
      int dst_KW = B->shape[3];

      int CPI = k_conv2d_cpi;  
      int CPO = k_conv2d_cpo;

      int GI      = dst_I / CPI;
      int GO      = dst_O / CPO;
      memset(dst_ptr, 0, sizeof(float) * dst_KW * dst_KH * dst_I * dst_O);

      for (int i=0; i < src_I; i++) {
        for (int o=0; o < src_O; o++) {
          for (int kh=0; kh<KH; kh++) {
            for (int kw=0; kw<KW; kw++) {
              int gi = i / CPI;
              int cpi = i % CPI;
              int go = o / CPO;
              int cpo = o % CPO;
              int in_addr = (o * KW * KH * src_I) + (i * KW * KH) + (kh * KW) + kw;
              int out_addr = (go * GI * CPO * CPI * dst_KH * dst_KW) + (gi * CPO * CPI * dst_KH * dst_KW) + 
                  (cpo * CPI * dst_KH * dst_KW) + (cpi * dst_KH * dst_KW) + (kh * dst_KW) + kw;
              dst_ptr[out_addr] = src_ptr[in_addr];
            }
          }
        }
      }
  }

void tensor_padded(Tensor *A, Tensor *B) {
  memset(B->ptr, 0, sizeof(float) * B->size);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++){
      B->ptr[i] = A->ptr[i];
  }
}

void get_batch_norm_values(int ochannels, Tensor *global_mean, Tensor *global_variance, Tensor* affine_g, Tensor* affine_b, Tensor* output) {
  memset(output->ptr, 0, sizeof(float) * output->size);
  // 0 (affine_b) 1 (affine_g) 2 (global_mean) 3 (global_variance)
  
  #pragma omp parallel for
  for (int i = 0; i < ochannels; i++){
      output->ptr[i*4]   = affine_b->ptr[i];
      output->ptr[i*4+1] = affine_g->ptr[i];
      output->ptr[i*4+2] = global_mean->ptr[i];
      output->ptr[i*4+3] = global_variance->ptr[i];
          printf("[out] %f %f %f %f\n", output->ptr[i*4], output->ptr[i*4+1], output->ptr[i*4+2], output->ptr[i*4+3]);
          printf("[inp] %f %f %f %f\n",affine_b->ptr[i] ,affine_g->ptr[i], global_mean->ptr[i], global_variance->ptr[i]);


  }
}
