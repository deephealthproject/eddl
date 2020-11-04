#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "xcl2.hpp"
//#include "/home/jorga20j/integration_eddl/eddl/fpga_kernels/test_fpga/test/src/xcl2.hpp"
//#include "/home/jomarm10/workspace/Vitis_Accel_Examples/common/includes/xcl2/xcl2.hpp"

using std::vector;

// CL
cl::Buffer buf;
cl::Context context;
cl::CommandQueue q;
cl::Program program;


#define SIZE 1024

static const int elements = 256;



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













void fpga_init(){ // initialize only once



}

void create_buffers() {




}

void fill(cl::Buffer *buf) {


}

void run() {


}

void run_cpu() {
}

void compare() {
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string binaryFile = argv[1];
  cl_int err;
  cl::Kernel kernel_relu;


  // size_t size_in_bytes = host_memory.size() * sizeof(int);

  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  std::cout << "Creating Context..." << std::endl;
  // The get_xil_devices will return vector of Xilinx Devices
  auto devices = xcl::get_xil_devices();
  auto device = devices[0];

  std::vector<float, aligned_allocator<float>> host_memory(elements, 42);
  // Creating Context and Command Queue for selected Device
  OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  std::cout << "Allocating and transferring data to " << device_name.c_str() << std::endl;

  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);

  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
  std::cout << "Device " << device_name.c_str() << ": program successful!" << std::endl;

  OCL_CHECK(err, kernel_relu = cl::Kernel(program,"k_relu", &err));
  std::cout << "Kernel sucessfully created" << std::endl ;

  size_t size_in_bytes = 4096*sizeof(float);
  // Allocate memory on the host and fill with random data.
  vector<float, aligned_allocator<float>> a(size_in_bytes);
  vector<float, aligned_allocator<float>> b(size_in_bytes);


  //-----------------------------
  // fill data vector with random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::cout << "Filling Tensor A with random values [-20.0, 30.0]" << std::endl ;
  for (int i = 0; i < SIZE; i++) {
    a[i] = dist(gen);
  }
  std::cout << "A[] = {" << std::endl;
  for (int i = 0; i < 20; i++) {
      std::cout << " " << a[i] << ",";
  }
  std::cout << " ...}" << std::endl ;

  //-----------------------------
  // THIS PAIR OF EVENTS WILL BE USED TO TRACK WHEN A KERNEL IS FINISHED WITH
  // THE INPUT BUFFERS. ONCE THE KERNEL IS FINISHED PROCESSING THE DATA, A NEW
  // SET OF ELEMENTS WILL BE WRITTEN INTO THE BUFFER.
  vector<cl::Event> kernel_events(1);
  vector<cl::Event> read_events(1);
  vector<cl::Event> write_events(1);
  cl::Buffer buffer_a, buffer_b;

  //-----------------------------
  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::cout << "Creating Buffers..." << std::endl;

  OCL_CHECK(err, buffer_a = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_in_bytes, &a[0], &err));
  OCL_CHECK(err, buffer_b = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR , size_in_bytes, &b[0], &err));

  // set kernel arguments
  OCL_CHECK(err, err = kernel_relu.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_relu.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_relu.setArg(2, (long int)SIZE));

  //-----------------------------
  // Copy input data to device global memory
  std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_a}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");

  //-----------------------------
  printf("Enqueueing NDRange kernel.\n");
  // This event needs to wait for the write buffer operations to complete
  // before executing. We are sending the write_events into its wait list to
  // ensure that the order of operations is correct.
  // Launch the Kernel
  std::vector<cl::Event> waitList;
  waitList.push_back(write_events[0]);
  OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_relu, 0, 1, 1, &waitList, &kernel_events[0]));
  set_callback(kernel_events[0], "ooo_queue");

  //-----------------------------
  // Copy Result from Device Global Memory to Host Local Memory
  std::cout << "Getting Results (Device to Host)..." << std::endl;
  std::vector<cl::Event> eventList;
  eventList.push_back(kernel_events[0]);
  // This operation only needs to wait for the kernel call. 
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, CL_MIGRATE_MEM_OBJECT_HOST, &eventList, &read_events[0]));
  set_callback(read_events[0], "ooo_queue");
  OCL_CHECK(err, err = read_events[0].wait());

  std::cout << "kernel returned" << std::endl ;
  std::cout << "  B [] = {" ;
   for (int i = 0; i < 10; i++) {
      std::cout << " " << b[i] << ",";
  }
  std::cout << " ... }" << std::endl;
  
  //-----------------------------
  // check received data
  std::cout << "Check kernel output, checking " << SIZE << " values"<< std::endl;
  {
    vector<float, aligned_allocator<float>> res_local(size_in_bytes);
    // perform kernel operation in host
    for (int i = 0; i < SIZE; i++ ) {
      if (a[i] < 0.0) res_local[i] = 0.0f;
      else            res_local[i] = a[i];
    }
    // compare data vectors
    int data_matches = 1;
    for (int i = 0; i < SIZE; i++) {
      if (res_local [i] != b[i]) {
        data_matches = 0;
        std::cout << "DATA MISMATCH    v_local[= " << i << "] = " << res_local[i] << "   !=   b[" << i << "] = " << b[i] << std::endl;
      }
    }

    if (data_matches) {
      std::cout << "" << std::endl;
      std::cout << "TEST PASSED" << std::endl;
    }
    else {
      std::cout << "" << std::endl;
      std::cout << "ERRORS DETECTED" << std::endl;
      std::cout << "TEST KO" << std::endl;
    }
  }

  //-----------------------------
  // HEY !!!!
  // It is necessary to release the resources, all of them,
  //  memories, buffers, kernels, programs,...

  // Wait for all of the OpenCL operations to complete
  std::cout << "Waiting..." << std::endl;
  OCL_CHECK(err, err = q.flush());
  OCL_CHECK(err, err = q.finish());

  //-----------------------------
  std::cout << "" << std::endl;
  std::cout << "All done" << std::endl;
  std::cout << "quit now" << std::endl;

  // exit
  return 0;
}
