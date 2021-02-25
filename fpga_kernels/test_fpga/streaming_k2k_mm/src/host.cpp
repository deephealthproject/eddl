// --------------------------------------------------------------------------------------------------------------
// FPGA kernels for EDDL Library - European Distributed Deep Learning Library.
// Version: 0.6
// copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
// Date: November 2020
// Authors: GAP Research Group (UPV)
//     José Flich Cardo
//     Jorge García Martínez
//     Izan Catalán Gallarch
//     Carles Hernández Luz
//
// contact: jflich@disca.upv.es
// All rights reserved

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <random>

#include <ap_fixed.h>
#include <sys/time.h>


// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"

using std::vector;

// data type
#define data_type float

// CL
cl::Buffer buf;
cl::Context context;
cl::CommandQueue q;
cl::Program program;
std::string binaryFile;


#define MULT 1
#define RELU 1  // Flag for ReLu activation. Active at high level

int W;
int H;
data_type value;


// buffers
data_type *data_in; //[  I * W * H        ]  __attribute__ ((__aligned__(16)));
data_type *data_out;     //[  O * W * H        ]  __attribute__ ((__aligned__(16)));
data_type *data_out_cpu; //[  O * W * H        ]  __attribute__ ((__aligned__(16)));

void allocate_buffers() {
  data_in = (data_type*)malloc(W * H * sizeof(data_type));
  data_out = (data_type*)malloc(W * H * sizeof(data_type));
  data_out_cpu = (data_type*)malloc(W * H * sizeof(data_type));
}

void parse_arguments(int argc, char **argv) {
  if (argc != 5) {
    printf("syntax:\n%s <XCLBIN File> <W> <H> <Value>\n", argv[0]);
    exit(1);
  }

  binaryFile = argv[1];
  W = atoi(argv[2]);
  H = atoi(argv[3]);
  value = std::stof(argv[4]);

}

void deallocate_buffers() {
  free(data_in);
  free(data_out);
  free(data_out_cpu);
}

void cpu_mult_relu() {

  int size = W * H;
  for(int i = 0; i< size; i++){
    if(MULT){
      data_out_cpu[i] = data_in[i]*value;
    }
    else{
      data_out_cpu[i] = data_in[i];
    }
    if(RELU){
      if(data_out_cpu[i] < 0) data_out_cpu[i] = 0;
    }
  }

}

void cpu_print_data_in() {
  printf("data in:\n");
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
      	// data_in pixel position
	      int addr_p =  (h * W) + w;
        printf("%6.2f ", float(data_in[addr_p]));
      }
      printf("\n");
    }
    printf("\n");
  }


void cpu_print_out() {
  printf("output: cpu (fpga)\n");
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
	      // data_out pixel position
        int addr_o = (h * W) + w;
        printf(" %6.2f (%6.2f) (diff %6.2f) | ", float(data_out_cpu[addr_o]), float(data_out[addr_o]), float(data_out_cpu[addr_o]-data_out[addr_o]));
      }
      printf("\n");
    }
  }

void check_result() {

  int error = 0;
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
      	// data_out pixel position
        int addr_o = (h * W) + w;
        if (fabs(float(data_out_cpu[addr_o]) - float(data_out[addr_o])) > 0.001) {
          printf("Results mismatch at h %d w %d: %6.2f %6.2f (diff %6.2f)\n", h, w, float(data_out_cpu[addr_o]), float(data_out[addr_o]), fabs(float(data_out_cpu[addr_o]-data_out[addr_o])));
          error = 1;
	  return;
	}
      }
    }
  if (!error) printf("results OK!\n"); else {
    printf("results differ:\n");
    //cpu_print_out();
  }
}


//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

// An event callback function that prints the operations performed by the OpenCL
// runtime.
void event_cb(cl_event event1, cl_int cmd_status, void *data) {
  cl_int err;
  cl_command_type command;
  cl::Event event(event1, true);
  OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_TYPE, &command));
  cl_int status;
  OCL_CHECK(err,
            err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status));
  const char *command_str;
  const char *status_str;
  switch (command) {
  case CL_COMMAND_READ_BUFFER:
    command_str = "buffer read";
    break;
  case CL_COMMAND_WRITE_BUFFER:
    command_str = "buffer write";
    break;
  case CL_COMMAND_NDRANGE_KERNEL:
    command_str = "kernel";
    break;
  case CL_COMMAND_MAP_BUFFER:
    command_str = "kernel";
    break;
  case CL_COMMAND_COPY_BUFFER:
    command_str = "kernel";
    break;
  case CL_COMMAND_MIGRATE_MEM_OBJECTS:
    command_str = "buffer migrate";
    break;
  default:
    command_str = "unknown";
  }
  switch (status) {
  case CL_QUEUED:
    status_str = "Queued";
    break;
  case CL_SUBMITTED:
    status_str = "Submitted";
    break;
  case CL_RUNNING:
    status_str = "Executing";
    break;
  case CL_COMPLETE:
    status_str = "Completed";
    break;
  }
  printf("[%s]: %s %s\n", reinterpret_cast<char *>(data), status_str,
         command_str);
  fflush(stdout);
}

// Sets the callback for a particular event
void set_callback(cl::Event event, const char *queue_name) {
  cl_int err;
  OCL_CHECK(err,
            err = event.setCallback(CL_COMPLETE, event_cb, (void *)queue_name));
}

//---------------------------------------------------------------------------------------------------------------------
////////MAIN FUNCTION//////////
int main(int argc, char **argv) {
  parse_arguments(argc, argv);

  printf("Test K2K MULT & RELU: \n");

  cl_int err;
  cl::Kernel kernel_mult_stream_k2k;
  cl::Kernel kernel_relu_stream_k2k;

  std::cout << "Creating Context..." << std::endl;
  auto devices = xcl::get_xil_devices();
  auto device = devices[0];
  OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);

  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
  std::cout << "Device " << device_name.c_str() << ": program successful!" << std::endl;

  OCL_CHECK(err, kernel_mult_stream_k2k = cl::Kernel(program,"krnl_stream_mult", &err));
  OCL_CHECK(err, kernel_relu_stream_k2k = cl::Kernel(program,"krnl_stream_relu", &err));
  std::cout << "Kernel sucessfully created" << std::endl ;

  size_t size_data_in_bytes = W * H * sizeof(data_type);
  size_t size_output_in_bytes = W * H * sizeof(data_type);

  // Allocate memory on the host and fill with random data.
  allocate_buffers();

  //-----------------------------
  // fill data vector with random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::cout << "Filling buffer with useful data" << std::endl;
  int addr = 0;
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
          data_in[addr] = dist(gen); //value;
          addr++;
      }
    }

  //
  // //-----------------------------
  // // THIS PAIR OF EVENTS WILL BE USED TO TRACK WHEN A KERNEL IS FINISHED WITH
  // // THE INPUT BUFFERS. ONCE THE KERNEL IS FINISHED PROCESSING THE DATA, A NEW
  // // SET OF ELEMENTS WILL BE WRITTEN INTO THE BUFFER.
  vector<cl::Event> kernel_events(2);
  vector<cl::Event> read_events(1);
  vector<cl::Event> write_events(1);
  cl::Buffer buffer_a;
  cl::Buffer buffer_b;

  //-----------------------------
  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::cout << "Creating Buffers..." << std::endl;

  cl_mem_ext_ptr_t data_in_ddr, data_out_ddr;
  //DATA IN
  data_in_ddr.flags  = XCL_MEM_DDR_BANK0;
  data_in_ddr.obj = data_in;
  data_in_ddr.param = 0;

  data_out_ddr.flags  = XCL_MEM_DDR_BANK3;
  data_out_ddr.obj = data_out;
  data_out_ddr.param = 0;

  OCL_CHECK(err, buffer_a = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY , size_data_in_bytes, &data_in_ddr, &err));
  OCL_CHECK(err, buffer_b = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY , size_output_in_bytes, &data_out_ddr, &err));



  //-----------------------------
  // Copy input data to device global memory
  // std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_a}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");


  // timint stats
  unsigned long long prof_time;
  struct timeval prof_t1;
  gettimeofday(&prof_t1, NULL);

  int enable_relu = RELU;
  int enable_mult = MULT;

  // set kernel arguments
  //MULT
  OCL_CHECK(err, err = kernel_mult_stream_k2k.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_mult_stream_k2k.setArg(2, H));
  OCL_CHECK(err, err = kernel_mult_stream_k2k.setArg(3, W));
  OCL_CHECK(err, err = kernel_mult_stream_k2k.setArg(4, value));
  OCL_CHECK(err, err = kernel_mult_stream_k2k.setArg(5, enable_mult));
  //RELU
  OCL_CHECK(err, err = kernel_relu_stream_k2k.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_relu_stream_k2k.setArg(2, H));
  OCL_CHECK(err, err = kernel_relu_stream_k2k.setArg(3, W));
  OCL_CHECK(err, err = kernel_relu_stream_k2k.setArg(4, enable_relu));


  // Launch the Kernel
 OCL_CHECK(err, err = q.enqueueTask(kernel_mult_stream_k2k, NULL, &kernel_events[0]));
 OCL_CHECK(err, err = q.enqueueTask(kernel_relu_stream_k2k, NULL, &kernel_events[1]));
 q.finish();

  // timing
  struct timeval prof_t2;
  gettimeofday(&prof_t2, NULL);
  prof_time = ((prof_t2.tv_sec - prof_t1.tv_sec) * 1000000) + (prof_t2.tv_usec - prof_t1.tv_usec);
  printf("Timing: %8lld usec\n", prof_time);

  // std::cout << "Getting Results (Device to Host)..." << std::endl;
  std::vector<cl::Event> eventList;
  // eventList.push_back(kernel_events[0]);
  // eventList.push_back(kernel_events[1]);
  // This operation only needs to wait for the kernel call.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, CL_MIGRATE_MEM_OBJECT_HOST, &eventList, &read_events[0]));
  set_callback(read_events[0], "ooo_queue");
  OCL_CHECK(err, err = read_events[0].wait());

  // Wait for all of the OpenCL operations to complete
  std::cout << "Waiting..." << std::endl;
  OCL_CHECK(err, err = q.flush());
  OCL_CHECK(err, err = q.finish());

  cl_ulong time_start, time_end;
  kernel_events[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
  kernel_events[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
  double diff = time_end-time_start;
  std::cout<< "TIME KERNEL MULT = " << (diff/1000000)<<" ms \n"<<std::endl;

  cl_ulong time_start1, time_end1;
  kernel_events[1].getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start1);
  kernel_events[1].getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end1);
  double diff1 = time_end1-time_start1;
  std::cout<< "TIME KERNEL RELU = " << (diff1/1000000)<<" ms \n"<<std::endl;

  std::cout << "computing mult relu in CPU..." << std::endl;

  cpu_print_data_in();
  cpu_mult_relu();
  cpu_print_out();
  check_result();

  //-----------------------------
  std::cout << "" << std::endl;
  std::cout << "All done" << std::endl;
  std::cout << "quit now" << std::endl;

  deallocate_buffers();

  // exit
  return 0;
}
