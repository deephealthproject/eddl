// --------------------------------------------------------------------------------------------------------------
// FPGA kernels for EDDL Library - European Distributed Deep Learning Library.
// Version: 0.6
// copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
// Date: December 2020
// Authors: GAP Research Group (UPV)
//     José Flich Cardo
//     Jorge García Martínez
//     Izan Catalán Gallarch
//     Carles Hernández Luz
//
// contact: jflich@disca.upv.es
// All rights reserved

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "xcl2.hpp"

#include <ap_fixed.h>
#include <sys/time.h>

using std::vector;

// data type
#define data_type float

#define SIZE  8  //size of the input vectors

// CL
cl::Buffer buf;
cl::Context context;
cl::CommandQueue q;
cl::Program program;
std::string binaryFile;

// buffers
data_type *data_in;
data_type *values;
data_type *data_out;
data_type *data_out_cpu;

void allocate_buffers() {
  data_in = (data_type*)malloc(SIZE * sizeof(data_type));
  values = (data_type*)malloc(SIZE * sizeof(data_type));
  data_out = (data_type*)malloc(SIZE * sizeof(data_type));
  data_out_cpu = (data_type*)malloc(SIZE * sizeof(data_type));
}

void parse_arguments(int argc, char **argv) {
  if (argc != 2) {
    printf("syntax:\n%s <XCLBIN File> \n", argv[0]);
    exit(1);
  }

  binaryFile = argv[1];

}

void deallocate_buffers() {
  free(data_in);
  free(values);
  free(data_out);
  free(data_out_cpu);
}

void cpu_mult_passive() {

  for (int i=0; i<SIZE; i++) data_out_cpu[i] = 0.f;
  //mult_passive
  for (int i=0; i<SIZE; i++){
    data_out_cpu[i] = data_in[i] * values[i];
  }

}

void cpu_print_data_in() {
  printf("data in: ");
  for (int i=0; i<SIZE; i++){
    printf("%6.2f ", data_in[i]);
  }
  printf("\n");
}

void cpu_print_values() {
  printf("values: ");
  for (int i=0; i<SIZE; i++){
    printf("%6.2f ", values[i]);
  }
  printf("\n");
}


void cpu_print_out() {
  printf("output: cpu (fpga)\n");
    for (int i=0; i<SIZE; i++){
      printf(" %10.6f (%10.6f) (diff %10.6f) | ", float(data_out_cpu[i]), float(data_out[i]), float(data_out_cpu[i]-data_out[i]));
    }
  printf("\n");
}

void check_result() {

  int error = 0;
  for (int i=0; i<SIZE; i++){
    if (fabs(float(data_out_cpu[i]) - float(data_out[i])) > 0.001) {
      printf("Results mismatch i = %d: %6.4f %6.4f (diff %6.4f)\n", i, float(data_out_cpu[i]), float(data_out[i]), fabs(float(data_out_cpu[i]-data_out[i])));
      error = 1;
	    return;
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

int main(int argc, char **argv) {

  parse_arguments(argc, argv);

  printf("Test mult passive\n");

  cl_int err;
  cl::Kernel kernel_mult_passive;

  std::cout << "Creating Context..." << std::endl;
  auto devices = xcl::get_xil_devices();
  auto device = devices[0];
  OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);

  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
  std::cout << "Device " << device_name.c_str() << ": program successful!" << std::endl;

  OCL_CHECK(err, kernel_mult_passive = cl::Kernel(program,"k_mult_passive", &err));
  std::cout << "Kernel sucessfully created" << std::endl ;

  size_t size_data_in_bytes = SIZE * sizeof(data_type);
  size_t size_output_in_bytes = SIZE * sizeof(data_type);
  size_t size_values_in_bytes = SIZE * sizeof(data_type);

  // Allocate memory on the host and fill with random data.
  allocate_buffers();

  //-----------------------------
  // fill data vector with random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::cout << "Filling buffer with useful data_in" << std::endl;
  for(int i=0; i<SIZE;i++){
    data_in[i] = dist(gen);
  }

  std::cout << "Filling values buffer with useful data" << std::endl;
  for (int i=0; i<SIZE; i++) {
    values[i] = dist(gen);
  }


  //-----------------------------
  // THIS PAIR OF EVENTS WILL BE USED TO TRACK WHEN A KERNEL IS FINISHED WITH
  // THE INPUT BUFFERS. ONCE THE KERNEL IS FINISHED PROCESSING THE DATA, A NEW
  // SET OF ELEMENTS WILL BE WRITTEN INTO THE BUFFER.
  vector<cl::Event> kernel_events(3);
  vector<cl::Event> read_events(3);
  vector<cl::Event> write_events(2);
  cl::Buffer buffer_a;
  cl::Buffer buffer_b;
  cl::Buffer buffer_v;


  //-----------------------------
  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::cout << "Creating Buffers..." << std::endl;

  OCL_CHECK(err, buffer_a = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_data_in_bytes, data_in, &err));
  OCL_CHECK(err, buffer_b = cl::Buffer(context, CL_MEM_WRITE_ONLY  | CL_MEM_USE_HOST_PTR , size_output_in_bytes, data_out, &err));
  OCL_CHECK(err, buffer_v = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_values_in_bytes, values, &err));


  //-----------------------------
  // Copy input data to device global memory
  // std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_a, buffer_v}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");

  // OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_v}, 0 /*0 means from host*/, NULL, &write_events[1]));
  // set_callback(write_events[1], "ooo_queue");


  // timint stats
  unsigned long long prof_time;
  struct timeval prof_t1;
  gettimeofday(&prof_t1, NULL);

  int size = SIZE;

  // set kernel arguments
  std::cout << "Sending args..." << std::endl;
  int arg = 0;
  OCL_CHECK(err, err = kernel_mult_passive.setArg(arg++, buffer_a));
  OCL_CHECK(err, err = kernel_mult_passive.setArg(arg++, buffer_b));
  OCL_CHECK(err, err = kernel_mult_passive.setArg(arg++, buffer_v));
  OCL_CHECK(err, err = kernel_mult_passive.setArg(arg++, size));



  //-----------------------------
  // printf("Enqueueing NDRange kernel.\n");
  // This event needs to wait for the write buffer operations to complete
  // before executing. We are sending the write_events into its wait list to
  // ensure that the order of operations is correct.
  // Launch the Kernel
  std::vector<cl::Event> waitList;
  waitList.push_back(write_events[0]);

  OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_mult_passive, 0, 1, 1, &waitList, &kernel_events[0]));
  set_callback(kernel_events[0], "ooo_queue");

  OCL_CHECK(err, err = kernel_events[0].wait());

  // std::cout << "Getting Results (Device to Host)..." << std::endl;
  std::vector<cl::Event> eventList;
  eventList.push_back(kernel_events[0]);
  // This operation only needs to wait for the kernel call.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, CL_MIGRATE_MEM_OBJECT_HOST, &eventList, &read_events[0]));
  set_callback(read_events[0], "ooo_queue");
  OCL_CHECK(err, err = read_events[0].wait());

  // Wait for all of the OpenCL operations to complete
  std::cout << "Waiting..." << std::endl;
  OCL_CHECK(err, err = q.flush());
  OCL_CHECK(err, err = q.finish());

  std::cout << "First time ..." << std::endl;
  cpu_print_data_in();
  cpu_print_values();
  cpu_mult_passive();
  cpu_print_out();

  check_result();


  OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_mult_passive, 0, 1, 1, &waitList, &kernel_events[1]));
  set_callback(kernel_events[1], "ooo_queue");

  OCL_CHECK(err, err = kernel_events[1].wait());
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, CL_MIGRATE_MEM_OBJECT_HOST, &eventList, &read_events[1]));

  set_callback(read_events[1], "ooo_queue");
  OCL_CHECK(err, err = read_events[1].wait());

  std::cout << "Waiting..." << std::endl;
  OCL_CHECK(err, err = q.flush());
  OCL_CHECK(err, err = q.finish());


  std::cout << "Second time ..." << std::endl;
  cpu_print_data_in();
  cpu_print_values();
  cpu_mult_passive();
  cpu_print_out();

  check_result();



  OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_mult_passive, 0, 1, 1, &waitList, &kernel_events[2]));
  set_callback(kernel_events[2], "ooo_queue");

  OCL_CHECK(err, err = kernel_events[2].wait());
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, CL_MIGRATE_MEM_OBJECT_HOST, &eventList, &read_events[2]));

  set_callback(read_events[2], "ooo_queue");
  OCL_CHECK(err, err = read_events[2].wait());

  std::cout << "Waiting..." << std::endl;
  OCL_CHECK(err, err = q.flush());
  OCL_CHECK(err, err = q.finish());


  std::cout << "Third time ..." << std::endl;
  cpu_print_data_in();
  cpu_print_values();
  cpu_mult_passive();
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
