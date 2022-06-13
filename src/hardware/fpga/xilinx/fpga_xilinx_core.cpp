/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

// Headers -------------------------------------------------------------------------------------------------------------------------------
//#include "eddl/hardware/fpga/xilinx/xcl2.hpp"      // OpenCL header   already included in fpga_hw
#include <vector>                           // Vectors
#include <math.h>                           // Math functions
#include <float.h>                          // Float operations
#include "eddl/tensor/tensor.h"             // EDDL Tensors
#include "eddl/descriptors/descriptors.h"   // EDDL Descriptors
#include "eddl/hardware/fpga/fpga_hw.h"     // FPGA enables of kernels, includes OpenCL headers and data types
#include <sys/time.h>                       // Time (for stats)
#include "eddl/hardware/cpu/cpu_tensor.h"   // CPU related function headers (cpu_transpose, cpu_copy, ...)
//#include <ap_fixed.h>                       // Aproximated precision fixed point support
//#include <ap_int.h>                         // Aproximated precision integer support
#include "eddl/profiling.h"                 // Profiling

#include "eddl/hardware/fpga/xilinx/fpga_xilinx_hw.h"

// Macros ---------------------------------------------------------------------------------------------------------------------------------
PROFILING_ENABLE_EXTERN(Precision_Conversion);
PROFILING_ENABLE_EXTERN(FPGA_READ);
PROFILING_ENABLE_EXTERN(FPGA_WRITE);

cl::Context               *context;                   // OpenCL context
std::vector<cl:: Device>   devices;                   // List of OpenCL devices
cl::Device                 device;                    // FPGA device
cl::CommandQueue          *q;                         // Command queue
cl::CommandQueue           com;                       // Command queue
cl::Program               *program;                   // Program

vector<cl::Event> kernel_events(MAX_KERNELS);         // Kernel events (completion)

cl::Kernel kernel_hlsinf[16];

// -------------------------------------------------------------------------------------------------------------------------------------------
// HLSinf related global variables

extern int hlsinf_filter_format;
extern int hlsinf_bias_format;
extern int hlsinf_input_format;
extern int hlsinf_output_format;
extern int hlsinf_cpi;
extern int hlsinf_cpo;
extern int hlsinf_num_kernels;
extern int hlsinf_ho_max;
extern int hlsinf_wo_max;
extern int hlsinf_max_rows;
extern std::string hlsinf_xclbin;
extern bool hlsinf_conv_support;
extern bool hlsinf_shift_support;
extern bool hlsinf_clip_support;
extern bool hlsinf_relu_support;
extern bool hlsinf_stm_support;
extern bool hlsinf_maxp_support;
extern bool hlsinf_avgp_support;
extern bool hlsinf_bn_support;
extern bool hlsinf_add_support;
extern bool hlsinf_upsize_support;
extern bool hlsinf_dense_support;

// -------------------------------------------------------------------------------------------------------------------------------------------
// OpenCL-related support functions ----------------------------------------------------------------------------------------------------------
//

// set_callback(). Sets the callback for a particular event in OpenCL
void set_callback(cl::Event event, const char *queue_name) {cl_int err; OCL_CHECK(err, err = event.setCallback(CL_COMPLETE, event_cb, (void *)queue_name));}

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

// -----------------------------------------------------------------------------------------------------------------------------------
// FPGA initialization and finalization functions

void close_fpga() {
  delete q;
  delete program;
  delete context;
}

// fpga_init()
// Initialices the device, sets up the kernels, prepares everything related to the FPGA device and support infrastructure
// This function must be called only once and at the begining of operations with the FPGA
void fpga_device_init() {
  #ifdef FPGA_DEBUG
  printf("initializing Xilinx FPGA\n");
  #endif

  cl_int      err;
  unsigned    fileBufSize;

  #ifdef FPGA_DEBUG
  std::cout << "Creating Context..." << std::endl;
  #endif
  
  devices = xcl::get_xil_devices();
  device = devices[0];

  OCL_CHECK(err, context = new cl::Context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, q = new cl::CommandQueue(*context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
    
  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  auto fileBuf = xcl::read_binary_file(hlsinf_xclbin);
  cl::Program::Binaries bins;

  bins = cl::Program::Binaries{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);

  OCL_CHECK(err, program = new cl::Program(*context, devices, bins, NULL, &err));

  #ifdef FPGA_DEBUG
  std::cout << "Device " << device_name.c_str() << ": program successful!" << std::endl;
  #endif

  // Now, we instatiate every possible kernel (enabled by the proper define)

  for (int k=0; k<hlsinf_num_kernels; k++) {
    char dummy[50];
    sprintf(dummy, "k_conv2D:{k_conv2D_%d}", k+1);
    OCL_CHECK(err, kernel_hlsinf[k] = cl::Kernel(*program, dummy, &err));
    std::cout << "Kernel sucessfully created" << std::endl ;
  }

  #ifdef FPGA_DEBUG
  printf("end of fpga_init\n");
  #endif
  
}


// ------------------------------------------------------------------------------------------------------------------------
// Copy operations
//


void *fpga_create_memory(long int size) {
  cl::Buffer *buffer;
  cl_int err;
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (creating memory in fpga size %d)\n", size);
  #endif

//  cl_mem_ext_ptr_t data_ddr;
//  data_ddr.flags  =  0 | XCL_MEM_TOPOLOGY;
//  data_ddr.obj = NULL;
//  data_ddr.param = 0;

  OCL_CHECK(err, buffer = new cl::Buffer(*context, CL_MEM_READ_WRITE /*| CL_MEM_EXT_PTR_XILINX*/, size, NULL /*&data_ddr*/, &err));
  return (void *)buffer;
}
void *fpga_create_memory(unsigned long flags, long int size) {
  return fpga_create_memory(size);
}



void fpga_copy_memory_to_fpga(void *ptr_cpu, void *ptr_fpga, long int size) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory to fpga: size %d, ptr_cpu %p)\n", size, ptr_cpu);
  #endif
  cl_int err;
  cl::Event blocking_event;
  cl::Buffer *cast_fpga_ptr = (cl::Buffer *)ptr_fpga;

  PROFILING_HEADER(FPGA_WRITE);
  OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*cast_fpga_ptr, CL_TRUE, 0, size, ptr_cpu, nullptr, &blocking_event));
  (*q).finish();
  PROFILING_FOOTER(FPGA_WRITE);
}

void fpga_copy_memory_to_fpga_and_format(void *ptr_cpu, void *ptr_fpga, long int size, int src_format, int dst_format) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory to fpga and format: size %d, ptr_cpu %p)\n", size, ptr_cpu);
  #endif
  cl_int err;
  cl::Event blocking_event;
  cl::Buffer *cast_ptr_fpga = (cl::Buffer *)ptr_fpga;

  if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_API8)) {
    PROFILING_HEADER(Precision_Conversion);
    float *src = (float*)ptr_cpu;
    ap_int<8> *cpu_buff = (ap_int<8> *)eddl_malloc(size * sizeof(ap_int<8>));
    for (int x = 0; x < size; x++) cpu_buff[x] = ap_int<8>(src[x]);
    PROFILING_FOOTER(Precision_Conversion);
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*cast_ptr_fpga, CL_TRUE, 0, size*sizeof(ap_int<8>), cpu_buff, nullptr, &blocking_event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_WRITE);
    free(cpu_buff);
  } else if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_API32)) {
    PROFILING_HEADER(Precision_Conversion);
    float *src = (float*)ptr_cpu;
    ap_int<32> *cpu_buff = (ap_int<32> *)eddl_malloc(size * sizeof(ap_int<32>));
    for (int x = 0; x < size; x++) {cpu_buff[x] = ap_int<32>(src[x]); /*printf("%f -> %f\n", src[x], float(cpu_buff[x]));*/}
    PROFILING_FOOTER(Precision_Conversion);
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*cast_ptr_fpga, CL_TRUE, 0, size*sizeof(ap_int<32>), cpu_buff, nullptr, &blocking_event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_WRITE);
    free(cpu_buff);
  } else if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_APF_8_4)) {
    PROFILING_HEADER(Precision_Conversion);
    float *src = (float*)ptr_cpu;
    ap_fixed<8,4,AP_RND_ZERO,AP_SAT> *cpu_buff = (ap_fixed<8,4,AP_RND_ZERO,AP_SAT> *)eddl_malloc(size * sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
    for (int x = 0; x < size; x++) {cpu_buff[x] = ap_fixed<8,4,AP_RND_ZERO,AP_SAT>(src[x]); /*if (size==64) printf("%f -> %f\n", src[x], float(cpu_buff[x]));*/}
    PROFILING_FOOTER(Precision_Conversion);
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*cast_ptr_fpga, CL_TRUE, 0, size*sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>), cpu_buff, nullptr, &blocking_event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_WRITE);
    free(cpu_buff);
  } else if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_APF_16_8)) {
    PROFILING_HEADER(Precision_Conversion);
    float *src = (float*)ptr_cpu;
    ap_fixed<16,8,AP_RND_ZERO,AP_SAT> *cpu_buff = (ap_fixed<16,8,AP_RND_ZERO,AP_SAT> *)eddl_malloc(size * sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
    for (int x = 0; x < size; x++) {cpu_buff[x] = ap_fixed<16,8,AP_RND_ZERO,AP_SAT>(src[x]); /*if (size==64) printf("%f -> %f\n", src[x], float(cpu_buff[x]));*/}
    PROFILING_FOOTER(Precision_Conversion);
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*cast_ptr_fpga, CL_TRUE, 0, size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>), cpu_buff, nullptr, &blocking_event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_WRITE);
    free(cpu_buff);
  } else if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_APF_32_16)) {
    PROFILING_HEADER(Precision_Conversion);
    float *src = (float*)ptr_cpu;
    ap_fixed<32,16> *cpu_buff = (ap_fixed<32,16> *)eddl_malloc(size * sizeof(ap_fixed<32,16>));
    for (int x = 0; x < size; x++) {cpu_buff[x] = ap_fixed<32,16>(src[x]); /*if (size==64) printf("%f -> %f\n", src[x], float(cpu_buff[x]));*/}
    PROFILING_FOOTER(Precision_Conversion);
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err= (*q).enqueueWriteBuffer(*cast_ptr_fpga, CL_TRUE, 0, size*sizeof(ap_fixed<32,16>), cpu_buff, nullptr, &blocking_event));
    (*q).finish();
    PROFILING_FOOTER(FPGA_WRITE);
    free(cpu_buff);
  } else {
    printf("copy with format not supported\n");
    exit(1);
  }
}

void fpga_copy_memory_from_fpga(void *ptr_fpga, void *ptr_cpu, long int size) {
  cl_int err;
  cl::Event event;
  cl::Buffer *cast_fpga_ptr = (cl::Buffer *)ptr_fpga;
  PROFILING_HEADER(FPGA_READ);
  OCL_CHECK(err, err = (*q).enqueueReadBuffer(*cast_fpga_ptr, CL_TRUE, 0, size, ptr_cpu, nullptr, &event));
  OCL_CHECK(err, err = event.wait());
  //(*q).finish();
  PROFILING_FOOTER(FPGA_READ);
}



// ----------------------------------------------------------------------------------------------------------------------------------------
// Support functions


// -----------------------------------------------------------------
// transform_nn
//
void fpga_transform_nn(Tensor *A, Tensor *B, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform) {
 _profile_fpga(_FPGA_TRANSFORM, 0);
 _debug_fpga_funcs("Transform");
 #ifdef FPGA_DEBUG
 printf("Transform\n");
 printf("  params: copy_cpu_to_fpga %d, copy_fpga_to_cpu %d, transform %d\n", copy_cpu_to_fpga, copy_fpga_to_cpu, transform);
#endif

  int CPI = hlsinf_cpi;

  if (!transform && copy_cpu_to_fpga) {

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
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(size_out);
      fpga_copy_memory_to_fpga(B->ptr, (cl::Buffer *)B->fpga_ptr, size_out);
    } else if (hlsinf_input_format == HLSINF_API8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_int<8>
      ap_int<8> *cpu_buff = (ap_int<8>*)malloc(B->size*sizeof(ap_int<8>));
      for (int x=0; x<B->size; x++) {
        ap_int<8> value = B->ptr[x];
        cpu_buff[x] = value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(B->size * sizeof(ap_int<8>));
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, B->size * sizeof(ap_int<8>));
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APUI8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_uint<8>
      unsigned char *cpu_buff = (unsigned char *)eddl_malloc(B->size*sizeof(unsigned char));
      for (int x=0; x<B->size; x++) {
        unsigned char value = B->ptr[x];
        cpu_buff[x] = value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(B->size*sizeof(ap_uint<8>));
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, B->size*sizeof(ap_uint<8>));
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APF_8_4) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_fixed<8,4,AP_RND_ZERO,AP_SAT>
      ap_fixed<8,4,AP_RND_ZERO,AP_SAT> *cpu_buff = (ap_fixed<8,4,AP_RND_ZERO,AP_SAT> *)eddl_malloc(B->size*sizeof(unsigned char));
      for (int x=0; x<B->size; x++) {
        ap_fixed<8,4,AP_RND_ZERO,AP_SAT> value = B->ptr[x];
        cpu_buff[x] = value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(B->size*sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, B->size*sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APF_16_8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_fixed<8,4,AP_RND_ZERO,AP_SAT>
      ap_fixed<16,8,AP_RND_ZERO,AP_SAT> *cpu_buff = (ap_fixed<16,8,AP_RND_ZERO,AP_SAT> *)eddl_malloc(B->size*sizeof(ap_fixed<16, 8>));
      for (int x=0; x<B->size; x++) {
        ap_fixed<16,8,AP_RND_ZERO,AP_SAT> value = B->ptr[x];
        cpu_buff[x] = value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(B->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, B->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APF_32_16) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_fixed<32,16>
      ap_fixed<32,16> *cpu_buff = (ap_fixed<32,16> *)eddl_malloc(B->size*sizeof(ap_fixed<32,16>));
      for (int x=0; x<B->size; x++) {
        ap_fixed<32,16> value = B->ptr[x];
        cpu_buff[x] = value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(B->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, B->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
      free(cpu_buff);
    } else {
      printf("Transform: input format not supported\n");
      exit(1);
    }
  } else if (transform && copy_cpu_to_fpga) {

    #ifdef FPGA_DEBUG
    printf("  input   "); _profile_cpu_tensor(A);
    #endif

    // transformation from CHW to GHWC (cpu to FPGA)
    // B_out, H_out and W_out assuned to be equal to B_in, H_in, W_in
    int B_in = A->shape[0]; int C_in = A->shape[1]; int H_in = A->shape[2]; int W_in = A->shape[3]; int C_out = B->shape[1];
    float *ptr_src = A->ptr; float *ptr_dst = B->ptr;

    int size_out = C_out * H_in * W_in * B_in * sizeof(float);
    memset(ptr_dst, 0, size_out);

    for (int b=0; b<B_in; b++) {
      for (int c=0; c<C_in; c++) {
	#pragma omp parallel for
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
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(size_out);
      fpga_copy_memory_to_fpga(B->ptr, (cl::Buffer *)B->fpga_ptr, size_out);
    } else if (hlsinf_input_format == HLSINF_API8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_int<8>
      ap_int<8> *cpu_buff = (ap_int<8>*)eddl_malloc(B->size*sizeof(ap_int<8>));
      for (int x=0; x<B->size; x++) {
        float value = B->ptr[x];
        cpu_buff[x] = ap_int<8>(value);
      }
      PROFILING_FOOTER(Precision_Conversion);
      size_out = C_out * H_in * W_in * B_in * sizeof(ap_int<8>);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(size_out);
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, size_out);
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APUI8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_uint<8>
      ap_uint<8> *cpu_buff = (ap_uint<8>*)eddl_malloc(B->size*sizeof(ap_uint<8>));
      for (int x=0; x<B->size; x++) {
        float value = B->ptr[x];
        cpu_buff[x] = ap_uint<8>(value);
      }
      PROFILING_FOOTER(Precision_Conversion);
      size_out = C_out * H_in * W_in * B_in * sizeof(ap_uint<8>);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(size_out);
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, size_out);
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APF_8_4) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_fixed<8,4,AP_RND_ZERO,AP_SAT>
      ap_fixed<8,4,AP_RND_ZERO,AP_SAT> *cpu_buff = (ap_fixed<8,4,AP_RND_ZERO,AP_SAT>*)eddl_malloc(B->size*sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
      for (int x=0; x<B->size; x++) {
        float value = B->ptr[x];
        cpu_buff[x] = ap_fixed<8,4,AP_RND_ZERO,AP_SAT>(value);
      }
      PROFILING_FOOTER(Precision_Conversion);
      size_out = C_out * H_in * W_in * B_in * sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(size_out);
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, size_out);
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APF_16_8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_fixed<8,4,AP_RND_ZERO,AP_SAT>
      ap_fixed<16,8,AP_RND_ZERO,AP_SAT> *cpu_buff = (ap_fixed<16,8,AP_RND_ZERO,AP_SAT>*)eddl_malloc(B->size*sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
      for (int x=0; x<B->size; x++) {
        float value = B->ptr[x];
        cpu_buff[x] = ap_fixed<16,8,AP_RND_ZERO,AP_SAT>(value);
      }
      PROFILING_FOOTER(Precision_Conversion);
      size_out = C_out * H_in * W_in * B_in * sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(size_out);
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, size_out);
      free(cpu_buff);
    } else if (hlsinf_input_format == HLSINF_APF_32_16) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_fixed<8,4,AP_RND_ZERO,AP_SAT>
      ap_fixed<32,16> *cpu_buff = (ap_fixed<32,16>*)eddl_malloc(B->size*sizeof(ap_fixed<32,16>));
      for (int x=0; x<B->size; x++) {
        float value = B->ptr[x];
        cpu_buff[x] = ap_fixed<32,16>(value);
      }
      PROFILING_FOOTER(Precision_Conversion);
      size_out = C_out * H_in * W_in * B_in * sizeof(ap_fixed<32,16>);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(size_out);
      fpga_copy_memory_to_fpga(cpu_buff, (cl::Buffer *)B->fpga_ptr, size_out);
      free(cpu_buff);
    } else {
      printf("Transform: input format not supported\n");
      exit(1);
    }
  } else if (!transform && copy_fpga_to_cpu) {

    float *ptr_dst = B->ptr;
    int num_elements = B->size;

    if (hlsinf_output_format == HLSINF_FP32) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(float));
      memcpy(B->ptr, A->ptr, num_elements * sizeof(float));
    } else if (hlsinf_output_format == HLSINF_API8) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_int<8>));
      PROFILING_HEADER(Precision_Conversion);
      for (int x=0; x < num_elements; x++) {ap_int<8> *ptr = (ap_int<8> *)A->ptr; float value = ptr[x]; B->ptr[x] = value;}
      PROFILING_FOOTER(Precision_Conversion);
    } else if (hlsinf_output_format == HLSINF_APUI8) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_uint<8>));
      PROFILING_HEADER(Precision_Conversion);
      for (int x=0; x < num_elements; x++) {ap_uint<8> *ptr = (ap_uint<8> *)A->ptr; float value = ptr[x]; B->ptr[x] = value;}
      PROFILING_FOOTER(Precision_Conversion);
    } else if (hlsinf_output_format == HLSINF_APF_8_4) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_fixed<8, 2,AP_RND_ZERO,AP_SAT>));
      PROFILING_HEADER(Precision_Conversion);
      for (int x=0; x < num_elements; x++) {ap_fixed<8, 2,AP_RND_ZERO,AP_SAT> *ptr = (ap_fixed<8, 2,AP_RND_ZERO,AP_SAT> *)A->ptr; float value = ptr[x]; B->ptr[x] = value;}
      PROFILING_FOOTER(Precision_Conversion);
    } else if (hlsinf_output_format == HLSINF_APF_16_8) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_fixed<16, 8,AP_RND_ZERO,AP_SAT>));
      PROFILING_HEADER(Precision_Conversion);
      for (int x=0; x < num_elements; x++) {ap_fixed<16, 8,AP_RND_ZERO,AP_SAT> *ptr = (ap_fixed<16, 8,AP_RND_ZERO,AP_SAT> *)A->ptr; float value = ptr[x]; B->ptr[x] = value;}
      PROFILING_FOOTER(Precision_Conversion);
    } else if (hlsinf_output_format == HLSINF_APF_32_16) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_fixed<32,16>));
      PROFILING_HEADER(Precision_Conversion);
      for (int x=0; x < num_elements; x++) {ap_fixed<32, 16> *ptr = (ap_fixed<32, 16> *)A->ptr; float value = ptr[x]; B->ptr[x] = value;}
      PROFILING_FOOTER(Precision_Conversion);
    } else {
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
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(float));
    } else if (hlsinf_output_format == HLSINF_API8) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_int<8>));
    } else if (hlsinf_output_format == HLSINF_APUI8) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_uint<8>));
    } else if (hlsinf_output_format == HLSINF_APF_8_4) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_fixed<8,4,AP_RND_ZERO,AP_SAT>));
    } else if (hlsinf_output_format == HLSINF_APF_16_8) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_fixed<16,8,AP_RND_ZERO,AP_SAT>));
    } else if (hlsinf_output_format == HLSINF_APF_32_16) {
      fpga_copy_memory_from_fpga((cl::Buffer *)A->fpga_ptr, A->ptr, num_elements * sizeof(ap_fixed<32,16>));
    } else {
      printf("Transform: output format not supported\n");
      exit(1);
    }

    #ifdef FPGA_DEBUG
    if (hlsinf_output_format == HLSINF_FP32) {printf("  input   "); _profile_cpu_tensor(A);}
    // in a different format we would need to move data to FP32
    #endif

    memset(ptr_dst, 0, size_dst);
    if (hlsinf_output_format == HLSINF_FP32) {
      for (int b=0; b<B_in; b++) {
        for (int c=0; c<C_in; c++) {
          #pragma omp parallel for
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
    } else if (hlsinf_output_format == HLSINF_API8) {
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
    } else if (hlsinf_output_format == HLSINF_APF_8_4) {
      PROFILING_HEADER(Precision_Conversion);
      for (int b=0; b<B_in; b++) {
        for (int c=0; c<C_in; c++) {
          for (int h=0; h<H_in; h++) {
            for (int w=0; w<W_in; w++) {
              int g = c / CPI;
              int cpi = c % CPI;
              int addr_src = (b * C_in * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
              int addr_dst = (b * C_out * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
              ap_fixed<8,4,AP_RND_ZERO,AP_SAT> *ptr_src = (ap_fixed<8,4,AP_RND_ZERO,AP_SAT> *)A->ptr;
              float value = float(ptr_src[addr_src]);
              ptr_dst[addr_dst] = value;
            }
          }
        }
      }
      PROFILING_FOOTER(Precision_Conversion);
    } else if (hlsinf_output_format == HLSINF_APF_16_8) {
      PROFILING_HEADER(Precision_Conversion);
      for (int b=0; b<B_in; b++) {
        for (int c=0; c<C_in; c++) {
          for (int h=0; h<H_in; h++) {
            for (int w=0; w<W_in; w++) {
              int g = c / CPI;
              int cpi = c % CPI;
              int addr_src = (b * C_in * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
              int addr_dst = (b * C_out * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
              ap_fixed<16,8,AP_RND_ZERO,AP_SAT> *ptr_src = (ap_fixed<16,8,AP_RND_ZERO,AP_SAT> *)A->ptr;
              float value = float(ptr_src[addr_src]);
              ptr_dst[addr_dst] = value;
            }
          }
        }
      }
      PROFILING_FOOTER(Precision_Conversion);
    } else if (hlsinf_output_format == HLSINF_APF_32_16) {
      PROFILING_HEADER(Precision_Conversion);
      for (int b=0; b<B_in; b++) {
        for (int c=0; c<C_in; c++) {
          for (int h=0; h<H_in; h++) {
            for (int w=0; w<W_in; w++) {
              int g = c / CPI;
              int cpi = c % CPI;
              int addr_src = (b * C_in * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
              int addr_dst = (b * C_out * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
              ap_fixed<32,16> *ptr_src = (ap_fixed<32,16> *)A->ptr;
              float value = float(ptr_src[addr_src]);
              ptr_dst[addr_dst] = value;
            }
          }
        }
      }
      PROFILING_FOOTER(Precision_Conversion);
    } else {
      printf("Transform: output format not supported\n");
      exit(1);
    }
    #ifdef FPGA_DEBUG
    printf("  output  "); _profile_cpu_tensor(B);
    #endif
  }

  _profile_fpga(_FPGA_TRANSFORM, 1);
}


#endif
