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

// S10MX included in common header file in stratix standalone development project
#include <CL/opencl.h>
#include <CL/cl_ext_intelfpga.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
// intel opencl support functions 
#include "eddl/hardware/fpga/intel/AOCLUtils/opencl.h"

using namespace aocl_utils;
// -- end of S10MX 

#include "eddl/hardware/fpga/intel/fpga_intel_hw.h"

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
cl_kernel          kernel_hlsinf[MAX_KERNELS][K_SUBKERNELS]; // this is kernel_hlsinf  WHY 16 SET "FORCED"
cl_event           kernel_events[MAX_KERNELS][K_SUBKERNELS];

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

extern bool hlsinf_intelkernel_version_implements_bn_add;

// -------------------------------------------------------------------------------------------------------------------------------------------
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


//  just for debug purposes
#define STRING_BUFFER_LEN 1024
// Helper functions to display parameters returned by OpenCL queries
void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   printf("%-40s = %lu\n", name, a);
}
void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   printf("%-40s = %u\n", name, a);
}
void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   printf("%-40s = %s\n", name, (a?"true":"false"));
}
void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[STRING_BUFFER_LEN]; 
   clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   printf("%-40s = %s\n", name, a);
}
void display_device_info( ) {

   printf("Querying device for info:\n");
   printf("========================\n");
   device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
   device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
   device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}

// -----------------------------------------------------------------------------------------------------------------------------------
size_t fpga_datatype_sizeof(int data_type) {
  ssize_t datatype_size;
  if (data_type == HLSINF_FP32 ) {
    datatype_size = sizeof(float);
  } else if (data_type == HLSINF_API8 ) {
    datatype_size = sizeof(int8_t);
  } else {
    printf("@fpga_datatype_sizeof. Intel FPGA Data (%d) type not supported\n", data_type);
    exit(1);
  }
  return datatype_size;
}

// -----------------------------------------------------------------------------------------------------------------------------------
float fpga_buffer_get_value_in_float(float *buf, int data_format, int index) {
  float v;

  if (data_format == HLSINF_FP32) {
    float *p = buf;
    v = p[index];
  } else if (data_format== HLSINF_API8) {
    int8_t *p = (int8_t *)buf;
    v =p[index];
  } else {
    printf("@fpga_buffer_get_value_in_float. Intel FPGA Data (%d) type not supported\n", data_format);
    exit(1);
  }

  return v;
}


// -----------------------------------------------------------------------------------------------------------------------------------
// FPGA initialization and finalization functions

void close_fpga() {
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
void fpga_device_init(int device_type) {
  #ifdef FPGA_DEBUG
  printf("initializing Intel FPGA\n");
  #endif

  cl_int      err;
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

  if (device_type == FPGA_PLATFORM_STRATIX_10MX_EB) {
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  }
  else if (device_type == FPGA_PLATFORM_STRATIX_10MX_EB_EMULATION){ 
    platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");;
  }
  else {
    printf("ERROR at fpga_device_init: UNKNOWN platform id %d\n", device_type);
  }

  err = 0;
  if (platform == NULL) {err = CL_DEVICE_NOT_FOUND;}
  OCL_CHECK(err, " -- ");

  // let's get the device id for the intel fpga, let's assume it is the first intel fpga in the list
  OCL_CHECK(err, err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, 0));


  #ifdef FPGA_DEBUG
  display_device_info();
  #endif

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
  OCL_CHECK(err, context = clCreateContext(0, 1, &device, &oclContextCallback, 0, &err));

  // let's create the command queues for each subkernel
  OCL_CHECK(err, q = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err););

  // let's load the kernel file for the stratix fpga(aocx)
  OCL_CHECK(err, bin_file = load_file(hlsinf_xclbin.c_str(), &bin_file_len, &err));

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
      #ifdef FPGA_DEBUG
      std::cout << "Creating ocl kernel # " << k << "  subkernel # " << l << "  subkernel name " << subkernel_names[l] << std::endl;
      #endif
      if (!hlsinf_intelkernel_version_implements_bn_add && ((l == K_BATCH_NORM_READER) || (l ==K_ADD_DATA_READER) || (l == K_BATCH_NORM) || (l == K_ADD_DATA)) )  {
        #ifdef FPGA_DEBUG
        std::cout << "    skipped. " <<"  INTEL FPGA kernel does not include BatchNormalization nor ADD suuport, skip kernel " << subkernel_names[l] <<" initialization!" << std::endl;
        #endif
        continue;
      }
      OCL_CHECK(err, kernel_hlsinf[k][l] = clCreateKernel(program, subkernel_names[l], &err));
    }
  }

  #ifdef FPGA_DEBUG
  printf("end of intel fpga device init\n");
  #endif
}


// ------------------------------------------------------------------------------------------------------------------------
// Copy operations
//


void fpga_destroy_memory(void *fpga_ptrI) {
  cl_mem tmp_ptr = (cl_mem)fpga_ptrI;
  //#ifdef FPGA_DEBUG_VERBOSE
  //printf("   destroy_memory buffer in FPGA\n");
  //#endif
  
  if(fpga_ptrI != nullptr) {
    clReleaseMemObject(tmp_ptr);
  }
}

unsigned long int dbg_global_buffers_size = 0;


//static uint jm10fpgacnt = 0;
void *fpga_create_memory(cl_mem_flags flags, long int size) {
  // cl_mem already is a pointer, so we can directly return a "cl_mem" type
  cl_mem buffer;
  cl_int err;
  #ifdef FPGA_DEBUG_VERBOSE
  dbg_global_buffers_size += size;
  printf("    (creating memory in fpga size %zu  (total %zu MB))\n", size, (dbg_global_buffers_size/1024/1024));
  #endif

  OCL_CHECK(err, buffer = clCreateBuffer(context, flags|CL_MEM_HETEROGENEOUS_INTELFPGA, size, nullptr, &err));

 // cl_mem_flags fl_ch = jm10fpgacnt == 0 ? CL_CHANNEL_1_INTELFPGA :
 //                      jm10fpgacnt == 1 ? CL_CHANNEL_2_INTELFPGA :
 //                      jm10fpgacnt == 2 ? CL_CHANNEL_3_INTELFPGA :
 //                      jm10fpgacnt == 3 ? CL_CHANNEL_4_INTELFPGA :
 //                      jm10fpgacnt == 4 ? CL_CHANNEL_5_INTELFPGA :
 //                      jm10fpgacnt == 5 ? CL_CHANNEL_6_INTELFPGA :
 //                                         CL_CHANNEL_7_INTELFPGA;
 // jm10fpgacnt = (jm10fpgacnt <= 5)? jm10fpgacnt++ : 0;
  // add to function call : cl_int mb_flags;  CL_MEM_READ_WRITE or CL_MEM_READ_ONLY or CL_MEM_WRITE_ONLY
  //OCL_CHECK(err, buffer = clCreateBuffer(context, flags|CL_MEM_HETEROGENEOUS_INTELFPGA|fl_ch, size, nullptr, &err));
  
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
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory to fpga and format: size %ld, ptr_cpu %p)\n", size, ptr_cpu);
  #endif
  cl_int err;
  cl_event blocking_event;

  if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_API8)) {
    PROFILING_HEADER(Precision_Conversion);
    float *src = (float*)ptr_cpu;
    int8_t *cpu_buff = (int8_t *)eddl_malloc(size * sizeof(int8_t));
    for (int x = 0; x < size; x++) cpu_buff[x] = (int8_t)src[x];
    PROFILING_FOOTER(Precision_Conversion);
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err = clEnqueueWriteBuffer(q, (cl_mem)ptr_fpga, CL_TRUE, 0, size*sizeof(int8_t), cpu_buff, 0, nullptr, &blocking_event));
    OCL_CHECK(err, err = clFinish(q));
    PROFILING_FOOTER(FPGA_WRITE);
    free(cpu_buff);
  } else if ((src_format == HLSINF_FP32) && (dst_format == HLSINF_API32)) {
    PROFILING_HEADER(Precision_Conversion);
    float *src = (float*)ptr_cpu;
    int32_t *cpu_buff = (int32_t *)eddl_malloc(size * sizeof(int32_t));
    for (int x = 0; x < size; x++) {cpu_buff[x] = (int32_t)src[x]; /*printf("%f -> %f\n", src[x], float(cpu_buff[x]));*/}
    PROFILING_FOOTER(Precision_Conversion);
    PROFILING_HEADER(FPGA_WRITE);
    OCL_CHECK(err, err = clEnqueueWriteBuffer(q, (cl_mem)ptr_fpga, CL_TRUE, 0, size*sizeof(int32_t), cpu_buff, 0, nullptr, &blocking_event));
    OCL_CHECK(err, err = clFinish(q));
    PROFILING_FOOTER(FPGA_WRITE);
    free(cpu_buff);
  } else {
    printf("copy with format not supported\n");
    exit(1);
  }

  // nota mental:
  // se puede optimizar
  //fpga_copy_memory_to_fpga( cpu_buff, ptr_fpga, size_formatted);
  //free (cpu_buff)

}

void fpga_copy_memory_from_fpga(void *ptr_fpga, void *ptr_cpu, long int size) {
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory from fpga: size %10ld, ptr_cpu %p)\n", size, ptr_cpu);
  #endif
  cl_int err;
  cl_event event;
  PROFILING_HEADER(FPGA_READ);
  OCL_CHECK(err, err = clEnqueueReadBuffer(q, (cl_mem)ptr_fpga, CL_TRUE, 0, size, ptr_cpu, 0, nullptr, &event));
  OCL_CHECK(err, err = clFinish(q));
  PROFILING_FOOTER(FPGA_READ);
  #ifdef FPGA_DEBUG_VERBOSE
  printf("    (copy memory from fpga          done for ptr_cpu %p)\n", ptr_cpu);
  #endif
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
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, size_out);
      fpga_copy_memory_to_fpga(B->ptr, B->fpga_ptr, size_out);
    } else if (hlsinf_input_format == HLSINF_API8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to ap_int<8>
      int8_t *cpu_buff = (int8_t *)eddl_malloc(B->size*sizeof(int8_t));
      for (int x = 0; x < B->size; x++) {
        int8_t value = B->ptr[x];
        cpu_buff[x] = value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, B->size * sizeof(int8_t));
      fpga_copy_memory_to_fpga(cpu_buff, B->fpga_ptr, B->size * sizeof(int8_t));
      free(cpu_buff);
    }
    else {
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
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, size_out);
      fpga_copy_memory_to_fpga(B->ptr, B->fpga_ptr, size_out);
    } else if (hlsinf_input_format == HLSINF_API8) {
      PROFILING_HEADER(Precision_Conversion);
      // We allocate a buffer to convert from floats to int8_t
      int8_t *cpu_buff = (int8_t *)eddl_malloc(B->size*sizeof(int8_t));
      #pragma omp parallel for
      for (int x=0; x<B->size; x++) {
        float value = B->ptr[x];
        cpu_buff[x] = (int8_t)value;
      }
      PROFILING_FOOTER(Precision_Conversion);
      size_out = C_out * H_in * W_in * B_in * sizeof(int8_t);
      if (B->fpga_ptr == NULL) B->fpga_ptr = fpga_create_memory(FPGA_CLMEM_READ_ONLY, size_out);
      fpga_copy_memory_to_fpga(cpu_buff, B->fpga_ptr, size_out);
      free(cpu_buff);
    }
    else {
      printf("Transform: input format not supported\n");
      exit(1);
    }
  } else if (!transform && copy_fpga_to_cpu) {
    float *ptr_dst = B->ptr;
    int num_elements = B->size;

    if (hlsinf_output_format == HLSINF_FP32) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(float));
      memcpy(B->ptr, A->ptr, num_elements * sizeof(float));
    } else if (hlsinf_output_format == HLSINF_API8) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(int8_t));
      PROFILING_HEADER(Precision_Conversion);
      #pragma omp parallel for
      for (int x=0; x < num_elements; x++) {int8_t *ptr = (int8_t *)A->ptr; float value = ptr[x]; B->ptr[x] = value;}
      PROFILING_FOOTER(Precision_Conversion);
    } 
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
    #ifdef FPGA_DEBUG
    printf("@fpga_intel_core: transformation from GHWC to CHW (FPGA to CPU)\n");
    #endif
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
    /*
    if (hlsinf_output_format == HLSINF_FP32) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(float));
    } else if (hlsinf_output_format == HLSINF_API8) {
      fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * sizeof(int8_t));
    } else {
      printf("Transform: output format not supported\n");
      exit(1);
    }
    */
    fpga_copy_memory_from_fpga(A->fpga_ptr, A->ptr, num_elements * fpga_datatype_sizeof(hlsinf_output_format));


    #ifdef FPGA_DEBUG
    if (hlsinf_output_format == HLSINF_FP32) printf("  input   "); _profile_cpu_tensor(A);
    // in a different format we would need to move data to FP32
    #endif

    memset(ptr_dst, 0, size_dst);

    if (hlsinf_output_format == HLSINF_FP32) {
    //printf("jm10 fpga_intel_core: transform and copy from fpga FP32\n");

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
    } else if (hlsinf_output_format == HLSINF_API8) {  
    //printf("jm10 fpga_intel_core: transform and copy from fpga API8\n");

      PROFILING_HEADER(Precision_Conversion);
      //#pragma omp parallel for 58
      for (int b=0; b<B_in; b++) {
        //#pragma omp parallel for 58
        for (int c=0; c<C_in; c++) {
          //#pragma omp parallel for 66
          for (int h=0; h<H_in; h++) {
            //#pragma omp parallel for
            for (int w=0; w<W_in; w++) {
              int g = c / CPI;
              int cpi = c % CPI; 
              int addr_src = (b * C_in * H_in * W_in) + (g * H_in * W_in * CPI) + (h * W_in * CPI) + (w * CPI) + cpi;
              int addr_dst = (b * C_out * H_in * W_in) + (c * H_in * W_in) + (h * W_in) + w;
              int8_t *ptr_src = (int8_t *)A->ptr;
              float value = float(ptr_src[addr_src]);
              ptr_dst[addr_dst] = (int8_t)value;
            }
          }
        }
      }
      PROFILING_FOOTER(Precision_Conversion);
    } 
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


#endif
