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
//--#include "eddl/hardware/fpga/xcl2.hpp"      // OpenCL header
//++ header de opencl de altera




#include <vector>                           // Vectors
#include <math.h>                           // Math functions
#include <float.h>                          // Float operations

// S10MX included in common header file in stratix standalone development project
#include <CL/opencl.h>
#include <CL/cl_ext_intelfpga.h>
#include "AOCLUtils/aocl_utils.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
// intel opencl support functions 
#include "eddl/hardware/fpga/AOCLUtils/opencl.h"
#include "eddl/hardware/fpga/icl_aux.h" 

using namespace aocl_utils;
// -- end of S10MX 



#include "eddl/tensor/tensor.h"             // EDDL Tensors
#include "eddl/descriptors/descriptors.h"   // EDDL Descriptors
#include "eddl/hardware/fpga/fpga_hw.h"     // FPGA enables of kernels
#include <sys/time.h>                       // Time (for stats)
#include "eddl/hardware/cpu/cpu_tensor.h"   // CPU related function headers (cpu_transpose, cpu_copy, ...)

//--#include <ap_fixed.h>                       // Aproximated precision fixed point support
//--#include <ap_int.h>                         // Aproximated precision integer support
//++ para altera no necesito ningún include con tipos de datos (al menos no uso ninguno de los includes que tiene)


#include "eddl/profiling.h"                 // Profiling

// Macros ---------------------------------------------------------------------------------------------------------------------------------
PROFILING_ENABLE_EXTERN(Precision_Conversion);
PROFILING_ENABLE_EXTERN(FPGA_READ);
PROFILING_ENABLE_EXTERN(FPGA_WRITE);

//--cl::Context               *context;                   // OpenCL context
//--std::vector<cl:: Device>   devices;                   // List of OpenCL devices
//--cl::Device                 device;                    // FPGA device
//--cl::CommandQueue          *q;                         // Command queue
//--cl::CommandQueue           com;                       // Command queue
//--cl::Program               *program;                   // Program
//--
//--vector<cl::Event> kernel_events(MAX_KERNELS);         // Kernel events (completion)
//--
//--cl::Kernel kernel_hlsinf[16];
// The set of simultaneous kernels

// An hlsinf_kernel consists of many opencl kernels
cl_platform_id              platform = NULL;
//std::vector <cl_device_id>  devices; // necessary?
cl_device_id                device = NULL;
cl_context                  context = NULL;
//cl_command_queue            q;
cl_program                  program = NULL;
//char                       *binaryFile;

//cl_command_queue queues[K_SUBERNELS];
cl_command_queue   q;
cl_kernel          kernel_hlsinf[16][K_SUBKERNELS]; // this is kernel_hlsinf  WHY 16 SET "FORCED"
cl_event           kernel_events[MAX_KERNELS][K_SUBKERNELS];

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
unsigned long long acc_time_fpga[_NUM_FPGA_FUNCS];  // accumulated ime of a kernel (function)

// profiling of FPGA resources being used
float mb_memory_needed_fpga;                        // Megabytes of memory needed for tensors in the FPGA



#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"


// OpenCL-related support functions ----------------------------------------------------------------------------------------------------------
//

unsigned char* load_file(const char *filename, size_t *size_ret, cl_int *err) {
  FILE *fp = fopen(filename, "rb");
  if(!fp) {
    printf("Failed to open the input file: %s.\n", filename);
    *err = CL_INVALID_BINARY;
    return NULL;
  }
  long size;
  unsigned char *buffer;

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  buffer = (unsigned char*)malloc(size);
  assert(buffer && "Malloc failed");
  size_t fread_sz = fread(buffer, size, 1, fp);
  if (fread_sz == 0) {
    printf("Failed to read from the AOCX file (fread).\n");
    fclose(fp);
    free(const_cast<unsigned char*>(buffer));
    return NULL;
  }
  fclose(fp);
  *size_ret = size;
  return buffer;
}

// set_callback(). Sets the callback for a particular event in OpenCL
void set_callback(cl_event event, const char *queue_name) {
  cl_int err; 
  OCL_CHECK(err, err = clSetEventCallback(event, CL_COMPLETE, event_cb, (void *)queue_name));  
}

// event_cb(). An event callback function that prints the operations performed by the OpenCL runtime
void event_cb(cl_event event1, cl_int cmd_status, void *data) {
  #ifdef FPGA_DEBUG
  
  cl_int err;
  cl_command_type command;
  cl_int status;

  //cl::Event event(event1, true);
  //OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_TYPE, &command));
  //OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status));
  
  OCL_CHECK(err, err = clGetEventInfo(event1, CL_EVENT_COMMAND_TYPE, sizeof(command), &command, NULL));
  OCL_CHECK(err, err = clGetEventInfo(event1, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL));


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
      case _FPGA_HLSINF                 : strcpy(name, "HLSinf"); break;
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
  #ifdef FPGA_XILINX
  else if (format_tensor == HLSINF_API32) size = T->size * sizeof(ap_int<32>);
  else if (format_tensor == HLSINF_API8) size = T->size * sizeof(ap_int<8>);
  else if (format_tensor == HLSINF_APUI8) size = T->size * sizeof(ap_uint<8>);
  #endif
  else {printf("format not supported in profile\n"); exit(1);}

  //if (T->fpga_ptr == NULL) T->fpga_ptr = fpga_create_memory(size);

  
   printf(KRED "JM10 warning!!\n" KNRM);
   printf(KYEL "@fpga_core->_profile_fpga_tensor replaced code to allocate aligned memory\n" KNRM);
  //float *buf = (float *)malloc(size);
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
    #ifdef FPGA_XILINX
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

void close_fpga() {
  //delete q;
  //delete program;
  //delete context;

  //for (int i = 0; i < K_SUBKERNELS; i++) {
  //  clReleaseCommandQueue(queues[i]);
  //}
  clReleaseCommandQueue(q);

  if(program) {
    clReleaseProgram(program);
    program = NULL;
  }
  if(context){
    clReleaseContext(context);
    context = NULL;
  }
}

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
  unsigned    fileBufSize;

  #ifdef FPGA_DEBUG
  std::cout << "Creating Context..." << std::endl;
  #endif
  
  cl_int status = 0;
  cl_int bin_status = 0;
  const unsigned char *bin_file;
  size_t bin_file_len = 0;
  char * device_name = NULL;

  //char *binaryFile = new char [hlsinf_xclbin.length()+1];
  //std::strcpy (binaryFile, hlsinf_xclbin.c_str());

  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  err = 0;
  if (platform == NULL) {err = CL_DEVICE_NOT_FOUND;}
  OCL_CHECK(err, " -- ");

  // let's get the device id for the intel fpga, let's assume it is the first intel fpga in the list
  OCL_CHECK(err, err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, 0));

  //// Start everything at NULL to help identify errors
  //for(int i = 0; i < K_SUBKERNELS; ++i){
  //  queues[i] = NULL;
  //}
  // Start everything at NULL to help identify errors
    q = NULL;

  for(int i = 0; i < 16; i++){
    for(int j = 0; j < K_SUBKERNELS; j++){
      kernel_hlsinf[i][j] = NULL;
    }
  }

  // let's create the context
  //OCL_CHECK(err, context = new cl::Context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, context = clCreateContext(0, 1, &device, &oclContextCallback, 0, &err));

  // let's create the command queues for each subkernel
  //OCL_CHECK(err, q = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
  //for(int i=0; i<K_SUBKERNELS; ++i) {
  //    OCL_CHECK(err, queues[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err););
  //}
  OCL_CHECK(err, q = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err););



  //std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  //auto fileBuf = xcl::read_binary_file(hlsinf_xclbin);
  // let's load the kernel binary file
  OCL_CHECK(err, bin_file = load_file(hlsinf_xclbin.c_str(), &bin_file_len, &err));

  //cl::Program::Binaries bins;
  //bins = cl::Program::Binaries{{fileBuf.data(), fileBuf.size()}};
  //devices.resize(1);
  //OCL_CHECK(err, program = new cl::Program(*context, devices, bins, NULL, &err));
  // let's create the program
  OCL_CHECK(err, program = clCreateProgramWithBinary(context, 1, &device, &bin_file_len, &bin_file, &bin_status, &err));
  OCL_CHECK(err, err = clBuildProgram(program, 1, &device, "", NULL, NULL));

  #ifdef FPGA_DEBUG
  //std::cout << "Device " << device_name.c_str() << ": program successful!" << std::endl;
  std::cout << "Device program successful!" << std::endl;
  #endif

  // Now, we instatiate every possible kernel (enabled by the proper define)
  for (int k=0; k<hlsinf_num_kernels; k++) {
    for(int l=0; l<K_SUBKERNELS; ++l) {
      OCL_CHECK(err, kernel_hlsinf[k][l] = clCreateKernel(program, subkernel_names[l], &err));
    }
  }

  #ifdef FPGA_DEBUG
  printf("end of fpga_init\n");
  #endif
}


// ------------------------------------------------------------------------------------------------------------------------
// Copy operations
//
void fpga_copy_to_fpga(float *nptr, Tensor *A, int cvt) {fpga_copy_to_fpga_good(nptr, A, cvt);}

void fpga_copy_to_fpga_good(float *nptr, Tensor *A, int cvt) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("\n");
  printf("FPGA_DEBUG: Copy CPU->FPGA. Addr: %p->%p. Size %4ld\n", nptr, A->fpga_ptr, A->size);
  #endif

  #ifdef PRECISION_CONVERSION
  // I do not remove this secction just in case I need it
  printf("JM10 - fpga_copy_to_fpga - precission conversion requested.\n");
  printf("Stratix kernel with FP32 should never reach this point.\n");
  printf(" abort execution\n");
  fflush(stdout);
  exit(1);
  if (cvt) {
    printf("ERROR Precission conversion not expected nor supported for Stratix kernels: FP32 support only\n");
    exit(1);
  } else {
    // regular copy from CPU to FPGA with no precision conversion
    cl_int err;
    cl_event event;
    cl_mem buf= (cl_mem) (A->fpga_ptr);
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err = clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, 0, NULL, &event));
    OCL_CHECK(err, err = clFinish(q));
    //OCL_CHECK(err, err = clEnqueueWriteBuffer(q, *buf, CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, 0, NULL, NULL));
    PROFILING_FOOTER(FPGA_WRITE);
  }
  #else
  // regular copy from CPU to FPGA with no precision conversion

  // option 1
  // #ifdef jm10_option1
  // cl_int err;
  // cl_event blocking_event; 
  // cl_mem buf= (cl_mem) A->fpga_ptr;
  // PROFILING_HEADER(FPGA_WRITE);
  // OCL_CHECK(err, err = clEnqueueWriteBuffer(q, *buf, CL_FALSE, 0, A->size*sizeof(fpga_data_type), nptr, 0, NULL, &blocking_event));
  // OCL_CHECK(err, err = clWaitForEvents(1, &blocking_event));
  // PROFILING_FOOTER(FPGA_WRITE);
  // OCL_CHECK(err, err = clReleaseEvent(blocking_event));
  // // jm10 do I need to run the finish(q)
  // // In such case, the command will be
  // //OCL_CHECK(err, err= clFinish(q));

  // // even easier, if we set the operations is blocking, since we wait
  // // option 2
  // //q...qwchich one?? -> simplify justone q for all kernels
  // #else
  cl_int err;
  cl_event event;
  cl_mem buf= (cl_mem) (A->fpga_ptr);
  PROFILING_HEADER(FPGA_WRITE);
  // clEnqueueWriteBuffer( command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
  OCL_CHECK(err, err = clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, 0, NULL, &event));
  OCL_CHECK(err, err = clFinish(q));
  
  //OCL_CHECK(err, err = clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, 0, NULL, NULL));
  PROFILING_FOOTER(FPGA_WRITE);
  // #endif
  //"jm10 -> fpga_core: should we release the cl_mem buf after writing data ?

  #endif
}

void fpga_copy_from_fpga_good(Tensor *A,float *nptr, int cvt) {fpga_copy_from_fpga(A, nptr, cvt);}

void fpga_copy_from_fpga(Tensor *A,float *nptr, int cvt) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("\n");
  printf("FPGA_DEBUG: Copy FPGA->CPU. Addr: %p->%p. Size %4ld\n", A->fpga_ptr, nptr, A->size);
  #endif

  #ifdef PRECISION_CONVERSION
  // We read from the FPGA to a temporal buffer and then convert the precision
  // I do not remove this secction just in case I need it
  printf("JM10 - fpga_copy_from_fpga - precission conversion requested.\n");
  printf("Stratix kernel with FP32 should never reach this point.\n");
  printf(" abort execution\n");
  fflush(stdout);
  exit(1);

  if (cvt) {
    _debug_fpga_funcs("Conversion (FPGA->CPU)");
    printf("ERROR Precission conversion not expected nor supported for Stratix kernels: FP32 support only\n");
    exit(1);
  } else {
    // regular copy from FPGA to CPU with no precision conversion
    cl_int err;
    cl_event event;
    PROFILING_HEADER(FPGA_READ);
    OCL_CHECK(err, err = clEnqueueReadBuffer(q, (cl_mem)(A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, 0,nullptr, &event));
    //OCL_CHECK(err, err = clWaitForEvents(1, &event));
    OCL_CHECK(err, err = clFinish(q));
    
    //OCL_CHECK(err, err = clEnqueueReadBuffer(q, (cl_mem)A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, nullptr, nullptr)
    PROFILING_FOOTER(FPGA_READ);
  }
  #else
  // regular copy from FPGA to CPU with no precision conversion
  cl_int err;
  cl_event event;
  PROFILING_HEADER(FPGA_READ);
  OCL_CHECK(err, err = clEnqueueReadBuffer(q, (cl_mem)(A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr,0, nullptr, &event));
  //OCL_CHECK(err, err = clWaitForEvents(1, &event));
  OCL_CHECK(err, err = clFinish(q));
  //OCL_CHECK(err, err = clEnqueueReadBuffer(q, (cl_mem)A->fpga_ptr), CL_TRUE, 0, A->size*sizeof(fpga_data_type), nptr, nullptr, nullptr);
  PROFILING_FOOTER(FPGA_READ);
  #endif
}

void fpga_destroy_memory(void *fpga_ptrI) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("   destroy_memory buffer in FPGA\n");
  #endif
  
  //if(fpga_ptrI != nullptr) {
  //  clReleaseMemObject(fpga_ptrI);
  //}
}

void *fpga_create_memory(cl_mem_flags flags, long int size) {
  // cl_mem already is a pointer, so we can directly return a "cl_mem" type
  cl_mem buffer;
  cl_int err;
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (creating memory in fpga size %zu)\n", size);
  #endif

  // add to function call : cl_int mb_flags;  CL_MEM_READ_WRITE or CL_MEM_READ_ONLY or CL_MEM_WRITE_ONLY
  OCL_CHECK(err, buffer = clCreateBuffer(context, flags, size, nullptr, &err));
  
  return buffer;
}


void fpga_copy_memory_to_fpga(void *ptr_cpu, void *ptr_fpga, long int size) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory to fpga: size %ld, ptr_cpu %p)\n", size, ptr_cpu);
  #endif
  cl_int err;
  cl_event blocking_event;
  PROFILING_HEADER(FPGA_WRITE);
  OCL_CHECK(err, err = clEnqueueWriteBuffer(q, (cl_mem)ptr_fpga, CL_TRUE, 0, size, ptr_cpu, 0, nullptr, &blocking_event));
  OCL_CHECK(err, err = clFinish(q));

  PROFILING_FOOTER(FPGA_WRITE);
}


void fpga_copy_memory_to_fpga_and_format(void *ptr_cpu, void *ptr_fpga, long int size, int src_format, int dst_format) {
//  #ifdef FPGA_DEBUG_VERBOSE
//  printf("    (copy memory to fpga and format: size %d, ptr_cpu %p)\n", size, ptr_cpu);
//  #endif
//  cl_int err;
//  cl::Event blocking_event;
//
//  if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_API8)) {
//    PROFILING_HEADER(Precision_Conversion);
//    float *src = (float*)ptr_cpu;
//    ap_int<8> *cpu_buff = (ap_int<8> *)malloc(size * sizeof(ap_int<8>));
//    for (int x = 0; x < size; x++) cpu_buff[x] = ap_int<8>(src[x]);
//    PROFILING_FOOTER(Precision_Conversion);
//    PROFILING_HEADER(FPGA_WRITE);
//    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*ptr_fpga, CL_TRUE, 0, size*sizeof(ap_int<8>), cpu_buff, nullptr, &blocking_event));
//    (*q).finish();
//    PROFILING_FOOTER(FPGA_WRITE);
//    free(cpu_buff);
//  } else if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_API32)) {
//    PROFILING_HEADER(Precision_Conversion);
//    float *src = (float*)ptr_cpu;
//    ap_int<32> *cpu_buff = (ap_int<32> *)malloc(size * sizeof(ap_int<32>));
//    for (int x = 0; x < size; x++) {cpu_buff[x] = ap_int<32>(src[x]); /*printf("%f -> %f\n", src[x], float(cpu_buff[x]));*/}
//    PROFILING_FOOTER(Precision_Conversion);
//    PROFILING_HEADER(FPGA_WRITE);
//    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*ptr_fpga, CL_TRUE, 0, size*sizeof(ap_int<32>), cpu_buff, nullptr, &blocking_event));
//    (*q).finish();
//    PROFILING_FOOTER(FPGA_WRITE);
//    free(cpu_buff);
//  } else {
    printf("copy with format not supported\n");
    exit(1);
//  }
}

void fpga_copy_memory_from_fpga(void *ptr_fpga, void *ptr_cpu, long int size) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory from fpga: size %ld, ptr_cpu %p)\n", size, ptr_cpu);
  #endif
  cl_int err;
  cl_event event;
  PROFILING_HEADER(FPGA_READ);
  OCL_CHECK(err, err = clEnqueueReadBuffer(q, (cl_mem)ptr_fpga, CL_TRUE, 0, size, ptr_cpu, 0, nullptr, &event));
  OCL_CHECK(err, err = clFinish(q));
  PROFILING_FOOTER(FPGA_READ);
}



// ----------------------------------------------------------------------------------------------------------------------------------------
// Support functions


#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
// -----------------------------------------------------------------
// transform_nn
//
void fpga_transform_nn(Tensor *A, Tensor *B, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform) {
 _profile_fpga(_FPGA_TRANSFORM, 0);
 _debug_fpga_funcs("Transform");
 #ifdef FPGA_DEBUG
 printf(KMAG "@fpga_core -> fpga_transform_nn copy_cpu_to_fpga %d, copy_fpga_to_cpu %d, transform %d\n" KNRM, copy_cpu_to_fpga, copy_fpga_to_cpu, transform);
#endif

  int CPI = hlsinf_cpi;

  if (!transform && copy_cpu_to_fpga) {
    
    printf(KMAG "dbug jm10 NOT transform and copy cpu to fpga\n" KNRM);


    #ifdef FPGA_DEBUG
    printf("  input   "); _profile_cpu_tensor(A);
    #endif

    // B_out, H_out and W_out assuned to be equal to B_in, H_in, W_in
    int B_in = A->shape[0]; int C_in = A->shape[1]; int H_in = A->shape[2]; int W_in = A->shape[3]; int C_out = B->shape[1];
    float *ptr_src = A->ptr; float *ptr_dst = B->ptr;

    int size_in = A->size * sizeof(float);
    int size_out = B->size * sizeof(float);
    memset(ptr_dst, 0, size_out);
    memcpy(ptr_dst, ptr_src, size_in);

    #ifdef FPGA_DEBUG
    printf("  output  "); _profile_cpu_tensor(B);
    #endif

    // copy to FPGA, source is in CPU and is in FP32, depending on the output format of HLSinf we convert if needed
    if (hlsinf_input_format == HLSINF_FP32) {
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(CL_MEM_READ_ONLY, size_out);
      fpga_copy_memory_to_fpga(B->ptr, (B->fpga_ptr), size_out);
    }
    #ifdef FPGA_XILINX
    else if (hlsinf_input_format == HLSINF_API8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_int<8>
      ap_int<8> *cpu_buff = (ap_int<8>*)malloc(B->size*sizeof(ap_int<8>));
      for (int x=0; x<B->size; x++) {
        ap_int<8> value = B->ptr[x];
        cpu_buff[x] = value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(CL_MEM_READ_ONLY, B->size * sizeof(ap_int<8>));
      fpga_copy_memory_to_fpga(cpu_buff, B->fpga_ptr, B->size * sizeof(ap_int<8>));
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APUI8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_uint<8>
      unsigned char *cpu_buff = (unsigned char *)malloc(B->size*sizeof(unsigned char));
      for (int x=0; x<B->size; x++) {
        unsigned char value = B->ptr[x];
        cpu_buff[x] = value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(CL_MEM_READ_ONLY, B->size*sizeof(ap_uint<8>));
      fpga_copy_memory_to_fpga(cpu_buff, B->fpga_ptr, B->size*sizeof(ap_uint<8>));
      free(cpu_buff);
    }
    #endif
    else
    {
      printf("Transform: input format not supported\n");
      exit(1);
    }


  } else if (transform && copy_cpu_to_fpga) {

    #ifdef FPGA_DEBUG
    printf("  input   "); _profile_cpu_tensor(A);
    #endif


    printf(KMAG "dbug jm10 transform and copy cpu to fpga\n" KNRM);

    // transformation from CHW to GHWC (cpu to FPGA)
    // B_out, H_out and W_out assuned to be equal to B_in, H_in, W_in
    int B_in = A->shape[0]; int C_in = A->shape[1]; int H_in = A->shape[2]; int W_in = A->shape[3]; int C_out = B->shape[1];
    float *ptr_src = A->ptr; float *ptr_dst = B->ptr;

    int size_out = C_out * H_in * W_in * B_in * sizeof(float);
    memset(ptr_dst, 0, size_out);

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

    #ifdef FPGA_DEBUG
    printf("  output  "); _profile_cpu_tensor(B);
    #endif

    // copy to FPGA, source is in CPU and is in FP32, depending on the output format of HLSinf we convert if needed
    if (hlsinf_input_format == HLSINF_FP32) {
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(CL_MEM_READ_ONLY, size_out);
      fpga_copy_memory_to_fpga(B->ptr, B->fpga_ptr, size_out);
    } 
    #ifdef FPGA_XILINX
    else if (hlsinf_input_format == HLSINF_API8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_int<8>
      ap_int<8> *cpu_buff = (ap_int<8>*)malloc(B->size*sizeof(ap_int<8>));
      for (int x=0; x<B->size; x++) {
        float value = B->ptr[x];
        cpu_buff[x] = ap_int<8>(value);
      }
      PROFILING_FOOTER(Precision_Conversion);
      size_out = C_out * H_in * W_in * B_in * sizeof(ap_int<8>);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(CL_MEM_READ_ONLY, size_out);
      fpga_copy_memory_to_fpga(cpu_buff, B->fpga_ptr, size_out);
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APUI8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_uint<8>
      ap_uint<8> *cpu_buff = (ap_uint<8>*)malloc(B->size*sizeof(ap_uint<8>));
      for (int x=0; x<B->size; x++) {
        float value = B->ptr[x];
        cpu_buff[x] = ap_uint<8>(value);
      }
      PROFILING_FOOTER(Precision_Conversion);
      size_out = C_out * H_in * W_in * B_in * sizeof(ap_uint<8>);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(CL_MEM_READ_ONLY, size_out);
      fpga_copy_memory_to_fpga(cpu_buff, B->fpga_ptr, size_out);
      free(cpu_buff);
    }
    #endif
    else {
      printf("Transform: input format not supported\n");
      exit(1);
    }
  } else if (!transform && copy_fpga_to_cpu) {
   
    printf(KMAG "dbug jm10 NOT transform and copy fpga to CPU\n" KNRM);

    float *ptr_dst = B->ptr;
    int num_elements = B->size;

    if (hlsinf_output_format == HLSINF_FP32) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(float));
      memcpy(B->ptr, A->ptr, num_elements * sizeof(float));
    } 
    #ifdef FPGA_XILINX
    else if (hlsinf_output_format == HLSINF_API8) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(ap_int<8>));
      PROFILING_HEADER(Precision_Conversion);
      for (int x=0; x < num_elements; x++) {ap_int<8> *ptr = (ap_int<8> *)A->ptr; float value = ptr[x]; B->ptr[x] = value;}
      PROFILING_FOOTER(Precision_Conversion);
    } else if (hlsinf_output_format == HLSINF_APUI8) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(ap_uint<8>));
      PROFILING_HEADER(Precision_Conversion);
      for (int x=0; x < num_elements; x++) {ap_uint<8> *ptr = (ap_uint<8> *)A->ptr; float value = ptr[x]; B->ptr[x] = value;}
      PROFILING_FOOTER(Precision_Conversion);
    }
    #endif
    else {
      printf("Transform: output format not supported\n");
      exit(1);
    }

    #ifdef FPGA_DEBUG
    printf("  input   "); _profile_cpu_tensor(A);
    printf("  output  "); _profile_cpu_tensor(B);
    #endif

  } else if (transform && copy_fpga_to_cpu) {

    // transformation from GHWC to CHW (FPGA to CPU)

    int B_in = A->shape[0];
    int C_in = A->shape[1];
    int H_in = A->shape[2];
    int W_in = A->shape[3];
    int C_out = B->shape[1];

    // B_out, H_out and W_out assuned to be equal to B_in, H_in, W_in

    //void *ptr_src = A->ptr;
    float *ptr_dst = B->ptr;
    int num_elements = C_out * H_in * W_in * B_in;
    int size_dst = C_out * H_in * W_in * B_in * sizeof(float);

    //printf("ptr_fpga %p ptr_dst %p size %d\n", A->fpga_ptr, B->ptr, size_dst);

    if (hlsinf_output_format == HLSINF_FP32) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(float));
    } 
    #ifdef FPGA_XILINX
    else if (hlsinf_output_format == HLSINF_API8) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(ap_int<8>));
    } else if (hlsinf_output_format == HLSINF_APUI8) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(ap_uint<8>));
    } 
    #endif
    else {
      printf("Transform: output format not supported\n");
      exit(1);
    }

    #ifdef FPGA_DEBUG
    if (hlsinf_output_format == HLSINF_FP32) printf("  input   "); _profile_cpu_tensor(A);
    // in a different format we would need to move data to FP32
    #endif

    memset(ptr_dst, 0, size_dst);

    if (hlsinf_output_format == HLSINF_FP32) {
      for (int b=0; b<B_in; b++) {
        for (int c=0; c<C_in; c++) {
          for (int h=0; h<H_in; h++) {
            for (int w=0; w<W_in; w++) {
              int g = c / CPI;
              int cpi = c % CPI; 
              int addr_src = (b * C_in * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
              int addr_dst = (b * C_out * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
              float *ptr_src = A->ptr;
              ptr_dst[addr_dst] = ptr_src[addr_src];
            }
          }
        }
      }
    }
    #ifdef FPGA_XILINX
    else if (hlsinf_output_format == HLSINF_API8) {  
      PROFILING_HEADER(Precision_Conversion);
      for (int b=0; b<B_in; b++) {
        for (int c=0; c<C_in; c++) {
          for (int h=0; h<H_in; h++) {
            for (int w=0; w<W_in; w++) {
              int g = c / CPI;
              int cpi = c % CPI; 
              int addr_src = (b * C_in * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
              int addr_dst = (b * C_out * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
              ap_int<8> *ptr_src = (ap_int<8> *)A->ptr;
              float value = float(ptr_src[addr_src]);
              ptr_dst[addr_dst] = value;
            }
          }
        }
      }
    } else if (hlsinf_output_format == HLSINF_APUI8) {  
      PROFILING_HEADER(Precision_Conversion);
      for (int b=0; b<B_in; b++) {
        for (int c=0; c<C_in; c++) {
          for (int h=0; h<H_in; h++) {
            for (int w=0; w<W_in; w++) {
              int g = c / CPI;
              int cpi = c % CPI; 
              int addr_src = (b * C_in * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
              int addr_dst = (b * C_out * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
              ap_uint<8> *ptr_src = (ap_uint<8> *)A->ptr;
              float value = float(ptr_src[addr_src]);
              ptr_dst[addr_dst] = value;
            }
          }
        }
      }


      PROFILING_FOOTER(Precision_Conversion);
    } 
    #endif
    else {
      printf("Transform: output format not supported\n");
      exit(1);
    } 
    #ifdef FPGA_DEBUG
    printf("  output  "); _profile_cpu_tensor(B);
    #endif
  }

  _profile_fpga(_FPGA_TRANSFORM, 1);
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

void fpga_write_buffer(char *file_name, void *ptr, int size, int data_size) {
  FILE *fd = fopen(file_name, "w");
  if (fd == NULL) {printf("Error, not able to open file for write\n"); exit(1);}
  void *buff = malloc(size * data_size);
  fpga_copy_memory_from_fpga(ptr, buff, size*data_size);
  fwrite(buff, data_size, size, fd);
  fclose(fd);
}

#endif
