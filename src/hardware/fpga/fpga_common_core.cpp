/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*
*
* version 1.1 add stratix support
*  major changes since Intel board uses C version of OpenCL 
*/

#ifdef cFPGA

// Headers -------------------------------------------------------------------------------------------------------------------------------
#include <vector>                           // Vectors
#include <math.h>                           // Math functions
#include <float.h>                          // Float operations

// openCL headers, vendor specific 
#ifdef cFPGA_VENDOR_XILINX
  #include "eddl/hardware/fpga/xilinx/fpga_xilinx_hw.h"
#endif

#ifdef cFPGA_VENDOR_INTEL
 #include "eddl/hardware/fpga/intel/fpga_intel_hw.h"
#endif


#include "eddl/tensor/tensor.h"             // EDDL Tensors
#include "eddl/descriptors/descriptors.h"   // EDDL Descriptors
#include "eddl/hardware/fpga/fpga_hw.h"     // FPGA enables of kernels, includes OpenCL headers and data types
#include <sys/time.h>                       // Time (for stats)
#include "eddl/hardware/cpu/cpu_tensor.h"   // CPU related function headers (cpu_transpose, cpu_copy, ...)
#include "eddl/profiling.h"                 // Profiling

// Macros ---------------------------------------------------------------------------------------------------------------------------------
PROFILING_ENABLE_EXTERN(Precision_Conversion);
PROFILING_ENABLE_EXTERN(FPGA_READ);
PROFILING_ENABLE_EXTERN(FPGA_WRITE);



// -------------------------------------------------------------------------------------------------------------------------------------------
// HLSinf related global variables

int hlsinf_filter_format;
int hlsinf_bias_format;
int hlsinf_input_format;
int hlsinf_output_format;
int hlsinf_cpi;
int hlsinf_cpo;
int hlsinf_num_kernels;
int hlsinf_ho_max;
int hlsinf_wo_max;
int hlsinf_max_rows;
std::string hlsinf_xclbin;
bool hlsinf_conv_support;
bool hlsinf_shift_support;
bool hlsinf_clip_support;
bool hlsinf_relu_support;
bool hlsinf_stm_support;
bool hlsinf_maxp_support;
bool hlsinf_avgp_support;
bool hlsinf_bn_support;
bool hlsinf_add_support;
bool hlsinf_upsize_support;
bool hlsinf_dense_support;

// -------------------------------------------------------------------------------------------------------------------------------------------
// Global variables for profiling
// Each kernel can be profiled (obtained the number of instances executed and the accumulated execution time)
//
int num_instances_fpga[_NUM_FPGA_FUNCS];            // number of instances a kernel (function) has been called
struct timeval time_ini_fpga[_NUM_FPGA_FUNCS];      // initial time of an instance for a kernel (function). Temporary variable
unsigned long long acc_time_fpga[_NUM_FPGA_FUNCS];  // accumulated time of a kernel (function)

// profiling of FPGA resources being used
float mb_memory_needed_fpga;                        // Megabytes of memory needed for tensors in the FPGA


// OpenCL-related support functions ----------------------------------------------------------------------------------------------------------
//

// _profile_fpga_funcname(). profiling function
void _profile_fpga_funcname(int i, char *name) {
  switch(i) {
      case _FPGA_HLSINF                : strcpy(name, "HLSinf"); break;
      default                          : strcpy(name, "?????"); break;
  }
}

// _profile_fpga(). Function to profile a kernel (function)
void _profile_fpga(int f_id, int end) {
  num_instances_fpga[f_id]++;
  if (!end) gettimeofday(&time_ini_fpga[f_id], NULL);
  else {
    timeval t1;
    gettimeofday(&t1, NULL);
    acc_time_fpga[f_id] += ((t1.tv_sec - time_ini_fpga[f_id].tv_sec) * 1000000) + (t1.tv_usec - time_ini_fpga[f_id].tv_usec);
  }
}

// profile_fpga_tensor(). Function to profile a tensor.
// It provides tensor information through the console
void _profile_fpga_tensor(const char str[], Tensor *T, int format_tensor) {
  #ifdef FPGA_DEBUG
  // We read the tensor from FPGA first
  int size;
  if (format_tensor == HLSINF_FP32) size = T->size * sizeof(float);
  #ifdef cFPGA_VENDOR_XILINX
  else if (format_tensor == HLSINF_API32) size = T->size * sizeof(ap_int<32>);
  else if (format_tensor == HLSINF_API8) size = T->size * sizeof(ap_int<8>);
  else if (format_tensor == HLSINF_APUI8) size = T->size * sizeof(ap_uint<8>);
  #endif
  else {printf("format not supported in profile\n"); exit(1);}

   std::string tmp_name = std::string(str) + "_profile";

  float *buf = (float *)eddl_malloc(size, tmp_name.c_str());
  fpga_copy_memory_from_fpga(T->fpga_ptr, buf, size);

  // Now we calculate statistics (min, max, avg) from the tensor
  float min = FLT_MAX;
  float max = -FLT_MAX;
  double sum = 0.f;
  double avg;
  for (int i=0; i<T->size; i++) {
    float v;
    if (format_tensor == HLSINF_FP32) {float *p = buf; v = p[i];
    }
    #ifdef cFPGA_VENDOR_XILINX
    else if (format_tensor == HLSINF_API32) {ap_int<32> *p = (ap_int<32> *)buf; v = float(p[i]);}
    else if (format_tensor == HLSINF_API8) {ap_int<8> *p = (ap_int<8> *)buf; v = float(p[i]);}
    else if (format_tensor == HLSINF_APUI8) {ap_uint<8> *p = (ap_uint<8> *)buf; v = p[i];}
    #endif
    else {printf("format not supported in profile\n"); exit(1);}
    if (v > max) max = v;
    if (v < min) min = v;
    sum += (double)v;
  }
  avg = sum / (double)T->size;

  // Now, we print the information (related tensor information and statistics of the tensor)
  printf("%s: - Tensor (fpga) ", str);
  printf(" size %10lu ", T->size);
  printf(" dims: ");
  printf(" %6d ", T->shape[0]);
  if (T->ndim>=2) printf(" x %6d ", T->shape[1]); else printf("          ");
  if (T->ndim>=3) printf(" x %6d ", T->shape[2]); else printf("          ");
  if (T->ndim>=4) printf(" x %6d ", T->shape[3]); else printf("          ");
  printf(" (cpu_ptr %p). ", T->ptr);
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
      int a = (d0 * T->shape[1] * T->shape[2] * T->shape[3]) + (d1 * T->shape[2] * T->shape[3]) + (d2 * T->shape[3]) + d3;
      printf("%f ", T->ptr[a]);
    }
    }
    }
    }
  }  else if(T->ndim==2) {
       for (int d0=0; d0<d1_max; d0++) {
       for (int d1=0; d1<d2_max; d1++) {
         int a = (d0 * T->shape[1]) + d1;
         printf("%f ", T->ptr[a]);
       }
       printf("\n\n");
    }

  } else if(T->ndim==1) {
    for (int d0=0; d0<T->shape[0]; d0++) {
      printf("%f ", T->ptr[d0]);
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

// -----------------------------------------------------------------------------------------------------------------------------------
// FPGA initialization and finalization functions
// fpga_init()
// Initialices the device, sets up the kernels, prepares everything related to the FPGA device and support infrastructure
// This function must be called only once and at the begining of operations with the FPGA
void fpga_init(int kernel_version, int kernel_subversion) {

  if (context!=NULL) {
    #ifdef FPGA_DEBUG
    printf("fpga_init function skipped, previous initialization done\n");
    //exit(1);
    return;
    #endif
  }

  #ifdef FPGA_DEBUG
  printf("initializing FPGA\n");
  #endif

  cl_int      err;

  // We need to instantiate the proper number of kernels, we also take the specifities of the kernels
  //
  // kernel versions:
  //             -------------------------------------------------------
  //             |                      Data format                    |
  //             |-----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  //     version | input  | conv filter | bias  | batch norm | output  | CPI x CPO | #kernels | HO max | WO max | max rows |   board             | xclbin             | Conv Type | Conv | Shift | Clip | ReLU | STM | MAXP | AVGP | BN | ADD | UPSIZE | Dense |
  //   ----------|-----------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  //   |   1.0   |  FP32  |   FP32      | FP32  |  FP32      |   FP32  |   4 x 4   |     2    |   256  |  1024  |    256   | Alveo U200          | hlsinf_v1.0.xclbin |   Direct  |   X  |       |   X  |  X   |  X  |  X   |  X   |  X |  X  |   X    |       |
  //   |   1.1   |  FP32  |   FP32      | FP32  |  FP32      |   FP32  |   4 x 4   |     2    |   256  |  1024  |    256   | Alveo U200          | hlsinf_v1.0.xclbin |   Direct  |   X  |       |   X  |  X   |  X  |  X   |  X   |  X |  X  |   X    |   X   |
  //   |   1.2   |  APUI8 |   API8      | API32 |  APUI8     |   APUI8 |   8 x 8   |     2    |   256  |  1024  |    256   | Alveo U200          | hlsinf_v1.1.xclbin |   Direct  |   X  |   X   |   X  |  X   |  X  |  X   |  X   |  X |  X  |   X    |       |
  //   |   1.3   |  APUI8 |   API8      | API32 |  APUI8     |   APUI8 |   8 x 8   |     2    |   256  |  1024  |    256   | Alveo U200          | hlsinf_v1.1.xclbin |   Direct  | Conv + Shift + Clip + ReLU       + {MaxP|AvgP} + BN + Add + Upsize |   X   |
  //   |   1.4   |  APUI8 |   API8      | API32 |  APUI8     |   APUI8 |  16 x 8   |     2    |   128  |  1024  |    128   | Alveo U200          | hlsinf_v1.2.xclbin |   Direct  | Conv + Shift + Clip + ReLU       + {MaxP|AvgP} + BN + Add + Upsize |       |
  //   |   1.5   |  APUI8 |   API8      | API32 |  APUI8     |   APUI8 |  16 x 8   |     2    |   128  |  1024  |    128   | Alveo U200          | hlsinf_v1.2.xclbin |   Direct  | Conv + Shift + Clip + ReLU       + {MaxP|AvgP} + BN + Add + Upsize |   X   |
  //   ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  //   |   2.0   |  FP32  |   FP32      | FP32  |  ----      |   FP32  |   4 x 4   |     1    |   256  |   256  |    256   | Stratix10 MX  | hlsinf_stratix_v2.0.aocx |   Direct  |   X  |   X   |   X  |  X   |     |  X   |  X   |    |     |        |       |
  //   |   2.1   |  FP32  |   FP32      | FP32  |  ----      |   FP32  |   8 x 8   |     1    |   256  |   256  |    256   | Stratix10 MX  | hlsinf_stratix_v2.1.aocx |   Direct  |   X  |   X   |   X  |  X   |     |  X   |  X   |    |     |        |       |
  //   |   2.2   |  FP32  |   FP32      | FP32  |  ----      |   FP32  |   8 x 8   |     1    |   256  |   512  |    256   | Stratix10 MX  | hlsinf_stratix_v2.2.aocx |   Direct  |   X  |   X   |   X  |  X   |     |  X   |  X   |    |     |        |       |
  //   ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  if ((kernel_version == 1) && (kernel_subversion == 0)) {
    hlsinf_filter_format = HLSINF_FP32; hlsinf_bias_format = HLSINF_FP32; hlsinf_input_format = HLSINF_FP32; hlsinf_output_format = HLSINF_FP32;
    hlsinf_cpi = 4; hlsinf_cpo = 4; hlsinf_num_kernels = 2;
    hlsinf_ho_max = 256; hlsinf_wo_max = 1024; hlsinf_max_rows = 256;
    hlsinf_xclbin = "hlsinf_v1.0.xclbin";
    hlsinf_conv_support = true; hlsinf_shift_support = false; hlsinf_clip_support = true; hlsinf_relu_support = true; hlsinf_stm_support = true; hlsinf_maxp_support = true; hlsinf_avgp_support = true; hlsinf_bn_support = true;
    hlsinf_add_support = true;  hlsinf_upsize_support = true;
    hlsinf_dense_support = false;
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v1.0: \n");
    printf("  Kernel configuration : FP32, CPIxCPO: 4x4, 2 kernels (hlsinf_v1.0.xclbin)\n");
    printf("  Platform             : Alveo U200 board\n");
    printf("  Supported layers     : CONV, CLIP, ReLU, SoftPlus, Tanh, Multiply Tensors, MaxPool, AvgPool, Batch Norm, Add Tensors, Upsize\n");
    printf("  Dense layer support  : No\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } else if ((kernel_version == 1) && (kernel_subversion == 1)) {
    hlsinf_filter_format = HLSINF_FP32; hlsinf_bias_format = HLSINF_FP32; hlsinf_input_format = HLSINF_FP32; hlsinf_output_format = HLSINF_FP32;
    hlsinf_cpi = 4; hlsinf_cpo = 4; hlsinf_num_kernels = 2;
    hlsinf_ho_max = 256; hlsinf_wo_max = 1024; hlsinf_max_rows = 256;
    hlsinf_xclbin = "hlsinf_v1.0.xclbin";
    hlsinf_conv_support = true; hlsinf_shift_support = false; hlsinf_clip_support = true; hlsinf_relu_support = true; hlsinf_stm_support = true; hlsinf_maxp_support = true; hlsinf_avgp_support = true; hlsinf_bn_support = true;
    hlsinf_add_support = true;  hlsinf_upsize_support = true;
    hlsinf_dense_support = true;
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v1.1: \n");
    printf("  Kernel configuration : FP32, CPIxCPO: 4x4, 2 kernels (hlsinf_v1.0.xclbin)\n");
    printf("  Platform             : Alveo U200 board\n");
    printf("  Supported layers     : CONV, CLIP, ReLU, SoftPlus, Tanh, Multiply Tensors, MaxPool, AvgPool, Batch Norm, Add Tensors, Upsize\n");
    printf("  Dense layer support  : Yes\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } else if ((kernel_version == 1) && (kernel_subversion == 2)) {
    hlsinf_filter_format = HLSINF_API8; hlsinf_bias_format = HLSINF_API32; hlsinf_input_format = HLSINF_APUI8; hlsinf_output_format = HLSINF_APUI8;
    hlsinf_cpi = 8; hlsinf_cpo = 8; hlsinf_num_kernels = 2;
    hlsinf_ho_max = 256; hlsinf_wo_max = 1024; hlsinf_max_rows = 256;
    hlsinf_xclbin = "hlsinf_v1.1.xclbin";
    hlsinf_conv_support = true; hlsinf_shift_support = true; hlsinf_clip_support = true; hlsinf_relu_support = true; hlsinf_stm_support = true; hlsinf_maxp_support = true; hlsinf_avgp_support = true; hlsinf_bn_support = true;
    hlsinf_add_support = true;  hlsinf_upsize_support = true;
    hlsinf_dense_support = false;
    printf("-----------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v1.2: \n");
    printf("  Kernel configuration : Mixed precission (weights apui<8>, bias<api32>, input apui<8>, output apui<8>), CPIxCPO: 8x8, 2 kernels (hlsinf_v1.1.xclbin)\n");
    printf("  Platform             : Alveo U200 board\n");
    printf("  Supported layers     : CONV, Shift, CLIP, ReLU, MaxPool, AvgPool, Batch Norm, Add Tensors, Upsize\n");
    printf("  Dense layer support  : No\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } else if ((kernel_version == 1) && (kernel_subversion == 3)) {
    hlsinf_filter_format = HLSINF_API8; hlsinf_bias_format = HLSINF_API32; hlsinf_input_format = HLSINF_APUI8; hlsinf_output_format = HLSINF_APUI8;
    hlsinf_cpi = 8; hlsinf_cpo = 8; hlsinf_num_kernels = 2;
    hlsinf_ho_max = 256; hlsinf_wo_max = 1024; hlsinf_max_rows = 256;
    hlsinf_xclbin = "hlsinf_v1.1.xclbin";
    hlsinf_conv_support = true; hlsinf_shift_support = true; hlsinf_clip_support = true; hlsinf_relu_support = true; hlsinf_stm_support = true; hlsinf_maxp_support = true; hlsinf_avgp_support = true; hlsinf_bn_support = true;
    hlsinf_add_support = true;  hlsinf_upsize_support = true;
    hlsinf_dense_support = true;
    printf("-----------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v1.3: \n");
    printf("  Kernel configuration : Mixed precission (weights apui<8>, bias<api32>, input apui<8>, output apui<8>), CPIxCPO: 8x8, 2 kernels (hlsinf_v1.1.xclbin)\n");
    printf("  Platform             : Alveo U200 board\n");
    printf("  Supported layers     : CONV, Shift, CLIP, ReLU, MaxPool, AvgPool, Batch Norm, Add Tensors, Upsize\n");
    printf("  Dense layer support  : Yes\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } else if ((kernel_version == 1) && (kernel_subversion == 4)) {
    hlsinf_filter_format = HLSINF_API8; hlsinf_bias_format = HLSINF_API32; hlsinf_input_format = HLSINF_APUI8; hlsinf_output_format = HLSINF_APUI8;
    hlsinf_cpi = 16; hlsinf_cpo = 8; hlsinf_num_kernels = 2;
    hlsinf_ho_max = 128; hlsinf_wo_max = 1024; hlsinf_max_rows = 128;
    hlsinf_xclbin = "hlsinf_v1.2.xclbin";
    hlsinf_conv_support = true; hlsinf_shift_support = true; hlsinf_clip_support = true; hlsinf_relu_support = true; hlsinf_stm_support = true; hlsinf_maxp_support = true; hlsinf_avgp_support = true; hlsinf_bn_support = true;
    hlsinf_add_support = true;  hlsinf_upsize_support = true;
    hlsinf_dense_support = false;
    printf("-----------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v1.4: \n");
    printf("  Kernel configuration : Mixed precission (weights apui<8>, bias<api32>, input apui<8>, output apui<8>), CPIxCPO: 16x8, 2 kernels (hlsinf_v1.2.xclbin)\n");
    printf("  Platform             : Alveo U200 board\n");
    printf("  Supported layers     : CONV, Shift, CLIP, ReLU, MaxPool, AvgPool, Batch Norm, Add Tensors, Upsize\n");
    printf("  Dense layer support  : No\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } else if ((kernel_version == 1) && (kernel_subversion == 5)) {
    hlsinf_filter_format = HLSINF_API8; hlsinf_bias_format = HLSINF_API32; hlsinf_input_format = HLSINF_APUI8; hlsinf_output_format = HLSINF_APUI8;
    hlsinf_cpi = 16; hlsinf_cpo = 8; hlsinf_num_kernels = 2;
    hlsinf_ho_max = 128; hlsinf_wo_max = 1024; hlsinf_max_rows = 128;
    hlsinf_xclbin = "hlsinf_v1.2.xclbin";
    hlsinf_conv_support = true; hlsinf_shift_support = true; hlsinf_clip_support = true; hlsinf_relu_support = true; hlsinf_stm_support = true; hlsinf_maxp_support = true; hlsinf_avgp_support = true; hlsinf_bn_support = true;
    hlsinf_add_support = true;  hlsinf_upsize_support = true;
    hlsinf_dense_support = true;
    printf("-----------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v1.4: \n");
    printf("  Kernel configuration : Mixed precission (weights apui<8>, bias<api32>, input apui<8>, output apui<8>), CPIxCPO: 16x8, 2 kernels (hlsinf_v1.2.xclbin)\n");
    printf("  Platform             : Alveo U200 board\n");
    printf("  Supported layers     : CONV, Shift, CLIP, ReLU, MaxPool, AvgPool, Batch Norm, Add Tensors, Upsize\n");
    printf("  Dense layer support  : Yes\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } else if ((kernel_version == 2) && (kernel_subversion == 0)) {
    hlsinf_filter_format = HLSINF_FP32; hlsinf_bias_format = HLSINF_FP32; hlsinf_input_format = HLSINF_FP32; hlsinf_output_format = HLSINF_FP32;
    hlsinf_cpi = 4; hlsinf_cpo = 4; hlsinf_num_kernels = 1;
    hlsinf_ho_max = 256; hlsinf_wo_max = 256; hlsinf_max_rows = 256;
    hlsinf_xclbin = "hlsinf_stratix_v2.0.aocx";
    hlsinf_conv_support = true;  hlsinf_shift_support = true; hlsinf_clip_support = true;
    hlsinf_relu_support = true;  hlsinf_stm_support = false;  hlsinf_maxp_support = true;
    hlsinf_avgp_support = true;  hlsinf_bn_support = false;   hlsinf_add_support = true;
    hlsinf_upsize_support = false; hlsinf_dense_support = false;
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v2.0: \n");
    printf("  Kernel configuration : FP32, FP32, CPIxCPO: %dx%d, WMAX %d  HMAX %d  %d kernel (%s)\n", hlsinf_cpi, hlsinf_cpo,hlsinf_wo_max, hlsinf_ho_max, hlsinf_num_kernels, hlsinf_xclbin.c_str());
    printf("  Platform             : Intel Stratix10 MX board\n");
    printf("  kernel freq          : 375.00 MHz\n");
    printf("  Supported layers     : CONV, Shift, CLIP, ReLU, MaxPool, AvgPool\n");
    printf("  Dense layer support  : No\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } else if ((kernel_version == 2) && (kernel_subversion == 1)) {
    hlsinf_filter_format = HLSINF_FP32; hlsinf_bias_format = HLSINF_FP32; hlsinf_input_format = HLSINF_FP32; hlsinf_output_format = HLSINF_FP32;
    hlsinf_cpi = 8; hlsinf_cpo = 8; hlsinf_num_kernels = 1;
    hlsinf_ho_max = 256; hlsinf_wo_max = 256; hlsinf_max_rows = 256;
    hlsinf_xclbin = "hlsinf_stratix_v2.1.aocx";
    hlsinf_conv_support = true;  hlsinf_shift_support = true; hlsinf_clip_support = true;
    hlsinf_relu_support = true;  hlsinf_stm_support = false;  hlsinf_maxp_support = true;
    hlsinf_avgp_support = true;  hlsinf_bn_support = false;   hlsinf_add_support = false;
    hlsinf_upsize_support = false;  hlsinf_dense_support = false;
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v2.1: \n");
    printf("  Kernel configuration : FP32, CPIxCPO: %dx%d, WMAX %d  HMAX %d  %d kernel (%s)\n", hlsinf_cpi, hlsinf_cpo,hlsinf_wo_max, hlsinf_ho_max, hlsinf_num_kernels, hlsinf_xclbin.c_str());
    printf("  Platform             : Intel Stratix10 MX board\n");
    printf("  kernel freq          : 268.75 MHz\n");
    printf("  Supported layers     : CONV, Shift, CLIP, ReLU, MaxPool, AvgPool\n");
    printf("  Dense layer support  : No\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } else if ((kernel_version == 2) && (kernel_subversion == 2)) {
    hlsinf_filter_format = HLSINF_FP32; hlsinf_bias_format = HLSINF_FP32; hlsinf_input_format = HLSINF_FP32; hlsinf_output_format = HLSINF_FP32;
    hlsinf_cpi = 8; hlsinf_cpo = 8; hlsinf_num_kernels = 1;
    hlsinf_ho_max = 256; hlsinf_wo_max = 512; hlsinf_max_rows = 256;
    hlsinf_xclbin = "hlsinf_stratix_v2.2.aocx";
    hlsinf_conv_support = true;  hlsinf_shift_support = true; hlsinf_clip_support = true;
    hlsinf_relu_support = true;  hlsinf_stm_support = false;  hlsinf_maxp_support = true;
    hlsinf_avgp_support = true;  hlsinf_bn_support = false;   hlsinf_add_support = false;
    hlsinf_upsize_support = false;  hlsinf_dense_support = false;
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
    printf("HLSinf accelerator v2.2: \n");
    printf("  Kernel configuration : FP32, CPIxCPO: %dx%d, WMAX %d  HMAX %d  %d kernel (%s)\n", hlsinf_cpi, hlsinf_cpo,hlsinf_wo_max, hlsinf_ho_max, hlsinf_num_kernels, hlsinf_xclbin.c_str());
    printf("  Platform             : Intel Stratix10 MX board\n");
    printf("  kernel freq          : 268.75 MHz\n");
    printf("  Supported layers     : CONV, Shift, CLIP, ReLU, MaxPool, AvgPool\n");
    printf("  Dense layer support  : No\n");
    printf("------------------------------------------------------------------------------------------------------------------------------\n");
  } 
  else {
    printf("Error, kernel version %d.%d not supported\n", kernel_version, kernel_subversion);
    exit(1);
  }

  fpga_device_init();
 
  #ifdef FPGA_DEBUG
  printf("end of fpga_init\n");
  #endif
}


// ------------------------------------------------------------------------------------------------------------------------
// Copy operations
//
// specific for each vendor, implemented in src/hardware/fpga/<vendor>/fpga_<vendor>_core.cpp

// ----------------------------------------------------------------------------------------------------------------------------------------
// Support functions


// -----------------------------------------------------------------
// transform_nn
//

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

      int CPI = hlsinf_cpi;  
      int CPO = hlsinf_cpo;

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
  
 // #pragma omp parallel for
  for (int i = 0; i < ochannels; i++){
      output->ptr[i*4]   = affine_b->ptr[i];
      output->ptr[i*4+1] = affine_g->ptr[i];
      output->ptr[i*4+2] = global_mean->ptr[i];
      output->ptr[i*4+3] = global_variance->ptr[i];
//          printf("[out] %f %f %f %f\n", output->ptr[i*4], output->ptr[i*4+1], output->ptr[i*4+2], output->ptr[i*4+3]);
//          printf("[inp] %f %f %f %f\n",affine_b->ptr[i] ,affine_g->ptr[i], global_mean->ptr[i], global_variance->ptr[i]);
  }
}


void dense_to_conv(float *ptr_src, int N, int M, float *ptr_dst, int I, int O, int KH, int KW) {

  // this function converts a weight matrix of NxM into a GIHWCPI organization, enabling
  // the multiplication operation as a convolution
  //

  memset(ptr_dst, 0, sizeof(float) * I * O * KH * KW);

  printf("ptr_src %p N %d M %d ptr_dst %p I %d O %d KH %d KW %d\n", ptr_src, N, M, ptr_dst, I, O, KH, KW);
  int CPI = hlsinf_cpi;  
  int CPO = hlsinf_cpo;
  int GI      = I / CPI;
  int GO      = O / CPO;

  for (int n=0; n<N; n++) {
    for (int m=0; m<M; m++) {
      int addr_src = n * M + m;
      int i = (n % CPI) + (CPI * (n/(9*CPI)));
      int o = m;
      int kh = (n % (CPI * KW * KH)) / (CPI * KW);
      int kw = ((n / CPI) % 3);

      int gi = i / CPI;
      int cpi = i % CPI;
      int go = o / CPO;
      int cpo = o % CPO;
//      printf("n %d m %d i %d kh %d kw %d o %d\n", n, m, i, kh, kw, o);
      int addr_dst = (go * GI * CPO * CPI * KH * KW) + (gi * CPO * CPI * KH * KW) + (cpo * CPI * KH * KW) + (cpi * KH * KW) + (kh * KW) + kw;
      ptr_dst[addr_dst] = ptr_src[addr_src];
    }
//    printf("fin para m = %d\n", m);
  }
//  printf("fin\n");
}


#ifdef WRITE_TENSORS_TO_FILE

void fpga_write_buffer(char *file_name, void *ptr, int size, int data_format) {
  FILE *fd = fopen(file_name, "w");
  if (fd == NULL) {printf("Error, not able to open file for write\n"); exit(1);}

  int data_size;
  if (data_format == HLSINF_API32) data_size = 4; else
  if (data_format == HLSINF_FP32) data_size = 4; else
  if (data_format == HLSINF_APUI8) data_size = 1; else
  if (data_format == HLSINF_API8) data_size = 1; else
  {printf("Error, no data format recognized\n"); exit(1);}
  printf("data_format %d data_size %d\n", data_format, data_size);
  
  void *buff = eddl_malloc(size * data_size);
  fpga_copy_memory_from_fpga(ptr, buff, size*data_size);
  float *buff1 = (float *)buff;
  fwrite(buff, data_size, size, fd);
  fclose(fd);
}
#endif

#endif
