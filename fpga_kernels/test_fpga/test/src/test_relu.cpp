#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */

#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "/home/jorga20j/test_fpga/test/src/xcl2.hpp"

// CL
cl::Buffer buf;
cl::Context context;
cl::CommandQueue q;
cl::Program program;
cl::Kernel kernel_relu;

#define SIZE 1024



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
 cl_int result_ready;
 // std::vector<int, aligned_allocator<int>> host_memory(elements, 42);
 // std::vector<int, aligned_allocator<int>> host_memory2(elements, 15);

 // size_t size_in_bytes = host_memory.size() * sizeof(int);

 // The get_xil_devices will return vector of Xilinx Devices
 auto devices = xcl::get_xil_devices();
 auto device = devices[0];

 // Creating Context and Command Queue for selected Device
 OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
 OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
 std::string device_name = device.getInfo<CL_DEVICE_NAME>();
 printf("Allocating and transferring data to %s\n", device_name.c_str());

 auto fileBuf = xcl::read_binary_file(binaryFile);
 cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
 devices.resize(1);
 OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

 float *a;
 size_t size_in_bytes = 4096*sizeof(float);
 posix_memalign((void **)&a,4096,SIZE*sizeof(float));

 for(int i=0; i<SIZE; i++) {
   a[i] = 0.1;
 }

 OCL_CHECK(err, cl::Buffer buffer_a(context, CL_MEM_COPY_HOST_PTR, size_in_bytes, a, &err));
 OCL_CHECK(err, cl::Buffer buffer_b(context, CL_MEM_WRITE_ONLY, size_in_bytes, nullptr, &err));

 OCL_CHECK(err, err = kernel_relu.setArg(0, buffer_a));
 OCL_CHECK(err, err = kernel_relu.setArg(1, buffer_b));
 OCL_CHECK(err, err = kernel_relu.setArg(2, SIZE));


 OCL_CHECK(err, err = q.enqueueTask(kernel_relu, NULL, &event));
 q.finish();

 OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b},CL_MIGRATE_MEM_OBJECT_HOST, NULL, &result_ready));
 result_ready.wait();
 printf("TEST PASSED\n");
  // exit
  return 0;
}
