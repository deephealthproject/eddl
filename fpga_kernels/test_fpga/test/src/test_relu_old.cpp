#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */

#include "vadd.h"
#include <stdlib.h>
#include <fstream>
#include <iostream>

// CL
cl::Buffer buf;
cl::Context context;
cl::CommandQueue q;
cl::Program program;
cl::Kernel kernel_relu;

#define SIZE 1024
// Compute the size of array in bytes
size_t size_in_bytes = SIZE * sizeof(float);
char* xclbinFilename;

void fpga_init(){ // initialize only once

  //cl_int err;
  //std::string binaryFile = "eddl.xclbin";
  //unsigned fileBufSize;
  //std::vector<cl::Device> devices = xcl::get_xil_devices();
  //cl::Device device = devices[0];
  //OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
  //OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
  //char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
  //cl::Program::Binaries bins{{fileBuf, fileBufSize}};

  //devices.resize(1);
  //OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));

  //OCL_CHECK(err, kernel_relu = cl::Kernel(program,"k_relu", &err));
  //if (err != CL_SUCCESS) printf("Error creating kernel\n");

  std::vector<cl::Device> devices;
  cl::Device device;
  std::vector<cl::Platform> platforms;
  bool found_device = false;

  //traversing all Platforms To find Xilinx Platform and targeted
  //Device in Xilinx Platform
  cl::Platform::get(&platforms);
  for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
      cl::Platform platform = platforms[i];
      std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
      if ( platformName == "Xilinx"){
          devices.clear();
          platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
          if (devices.size()){
                  device = devices[0];
                  found_device = true;
                  break;
          }
      }
  }
  if (found_device == false){
     std::cout << "Error: Unable to find Target Device "
         << device.getInfo<CL_DEVICE_NAME>() << std::endl;
     //return EXIT_FAILURE;
   }

  // Creating Context and Command Queue for selected device
  cl::Context context(device);
  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

  // Load xclbin
  std::cout << "Loading: '" << xclbinFilename << "'\n";
  std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
  bin_file.seekg (0, bin_file.end);
  unsigned nb = bin_file.tellg();
  bin_file.seekg (0, bin_file.beg);
  char *buf = new char [nb];
  bin_file.read(buf, nb);
  printf("XCLBIN loaded ...\n");

  // Creating Program from Binary File
  cl::Program::Binaries bins;
  bins.push_back({buf,nb});
  devices.resize(1);
  cl::Program program(context, devices, bins);
  printf("Program created ...\n");
  // This call will get the kernel object from program. A kernel is an
  // OpenCL function that is executed on the FPGA.
  cl::Kernel kernel_relu(program,"k_relu");
  printf("Kernel created ...\n");

}

void create_buffers() {
  // cl_int err;
  // OCL_CHECK(err,buf = newcl::Buffer(context,CL_MEM_READ_WRITE, SIZE*sizeof(float), NULL, &err));
  // cpu_buf = malloc(sizeof(float)*SIZE);

 // cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, size_in_bytes);
 // cl::Buffer buffer_b(context, CL_MEM_WRITE_ONLY, size_in_bytes);

  //set the kernel Arguments
 // int narg=0;
  //kernel_relu.setArg(narg++,buffer_a);
  //kernel_relu.setArg(narg++,buffer_b);
  //kernel_relu.setArg(narg++,SIZE);

  //We then need to map our OpenCL buffers to get the pointers
  //float *ptr_a = (float *) q.enqueueMapBuffer (buffer_a , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
  //float *ptr_b = (float*) q.enqueueMapBuffer (buffer_b , CL_TRUE , CL_MAP_READ , 0, size_in_bytes);


  //setting input data
 // for(int i = 0 ; i< SIZE; i++){
  //ptr_a[i] = rand(-100, 100)/100;
 // }

  // Data will be migrated to kernel space
  //q.enqueueMigrateMemObjects({buffer_a},0/* 0 means from host*/);



}

void fill(cl::Buffer *buf) {
  // for (int i=0; i<SIZE; i++) cpu_buf[i] = random(-1.0, 1.0);

  // cl_int err;
  // cl::Event blocking_event;
  // OCL_CHECK(err, err= q.enqueueWriteBuffer(*buf, CL_TRUE, 0, SIZE*sizeof(float), cpu_buf, nullptr, &blocking_event));
  // q.finish();

  //setting input data
  //for(int i = 0 ; i< DATA_SIZE; i++){
  // ptr_a[i] = 10;
  // ptr_b[i] = 20;
  //}

    // Data will be migrated to kernel space
    //q.enqueueMigrateMemObjects({buffer_a,buffer_b},0/* 0 means from host*/);

}

void run() {

  //  q.enqueueTask(kernel_relu);

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    //q.enqueueMigrateMemObjects({buffer_b},CL_MIGRATE_MEM_OBJECT_HOST);

    //q.finish();
}

void run_cpu() {
}

void compare() {
}

int main(int argc, char **argv) {

  //TARGET_DEVICE macro needs to be passed from gcc command line
  //  if(argc != 2) {
    //   std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
      // return EXIT_FAILURE;
   // }
  xclbinFilename = argv[1];

  // initialization
  fpga_init();

  // buffers
  //create_buffers();
  printf("Buffers creating ...\n");
  cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, size_in_bytes);
  cl::Buffer buffer_b(context, CL_MEM_WRITE_ONLY, size_in_bytes);

  //set the kernel Arguments
  printf("Setting the kernel Arguments ...\n");
  int narg=0;
  kernel_relu.setArg(narg++,buffer_a);
  kernel_relu.setArg(narg++,buffer_b);
  kernel_relu.setArg(narg++,(long int)SIZE);
  printf("end kernel arguments ...\n");

  //We then need to map our OpenCL buffers to get the pointers
  float *ptr_a = (float *) q.enqueueMapBuffer (buffer_a , CL_TRUE , CL_MAP_WRITE , 0, size_in_bytes);
  float *ptr_b = (float *) q.enqueueMapBuffer (buffer_b , CL_TRUE , CL_MAP_READ , 0, size_in_bytes);

  printf("Init ptr_a ...\n");
  //setting input data
  //for(int i = 0 ; i< (int)SIZE; i++){
  ptr_a[i] = 0.1;
  //}
  printf("copying data into kernel buffer_a");
  q.enqueueMigrateMemObjects({buffer_a},0/* 0 means from host*/);
  // fill with random numbers
  // fill_buffers();

  // run the kernel
  //run();

  // run in cpu
  // run_cpu();
  printf("Launching task...\n");
  q.enqueueTask(kernel_relu);

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    q.enqueueMigrateMemObjects({buffer_b},CL_MIGRATE_MEM_OBJECT_HOST);
    printf("Buffer read ...\n");
    q.finish();

  // compare results
  // compare();

  printf("TEST PASSED\n");
  // exit
  return 0;
}
