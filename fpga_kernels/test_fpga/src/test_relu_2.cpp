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
cl::Context context;
cl::Kernel kernel_ut;
cl::CommandQueue q;
cl::Program program;



//---------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {



  std::default_random_engine generator;
  std::normal_distribution<float> distribution(-1.0,1.0);


  std::string binaryFile = argv[1];
  cl_int err;
  cl::Event  event;
  cl::Event write_event;
  cl::Event result_ready;



  //fpga_init(binaryFile);

  //---------------------------------------------------------------------------
  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  std::cout << "Creating Context..." << std::endl;
  // The get_xil_devices will return vector of Xilinx Devices
  auto devices = xcl::get_xil_devices();
  auto device = devices[0];

//  std::vector<float, aligned_allocator<float>> host_memory(elements, 42);
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

  OCL_CHECK(err, cl::Kernel kernel_ut(program,"k_relu", &err));
  std::cout << "Kernel sucessfully created" << std::endl ;

  int filas = 128;
  int columnas = 1024;

  int size_ab = filas*columnas;
  size_t size = 128*1024;
  size_t size_a_in_bytes = size * sizeof(float);
  size_t size_b_in_bytes = size * sizeof(float);

  // Allocate memory on the host
  vector<float, aligned_allocator<float>> a(size, 0);
  vector<float, aligned_allocator<float>> b(size, 0);

  float A_CPU[128*1024];
  float B_CPU[128*1024];


  //Vector A
  // printf("Vector A init \n");
  for (int i = 0; i < size_ab; i++) {
        a[i] = distribution(generator);
        A_CPU[i] = a[i];
  }

  // printf("a: ");
  // for (int i = 0; i < size_ab; i++) {
  //   printf("%f ", a[i]);
  // }
  // printf("\n");
  //
  // printf("A_CPU:  ");
  // for (int i = 0; i < size_ab; i++) {
  //   printf("%f ", A_CPU[i]);
  // }
  // printf("\n");

  //Vector B
  // printf("Vector B init \n");
  for (int i = 0; i < size_ab; i++) {
        b[i] = 0;
        B_CPU[i]=0;
  }

  // for (int i = 0; i < size_ab; i++) {
  //   printf("%f ", b[i]);
  //   }
  //   printf("\n");

  //-----------------------------
  // THIS PAIR OF EVENTS WILL BE USED TO TRACK WHEN A KERNEL IS FINISHED WITH
  // THE INPUT BUFFERS. ONCE THE KERNEL IS FINISHED PROCESSING THE DATA, A NEW
  // SET OF ELEMENTS WILL BE WRITTEN INTO THE BUFFER.
  // vector<cl::Event> kernel_events(1);
  // vector<cl::Event> read_events(1);
  // vector<cl::Event> write_events(1);
  cl::Buffer buffer_a, buffer_b;

  //-----------------------------
  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::cout << "Creating Buffers..." << std::endl;

  OCL_CHECK(err, buffer_a = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_a_in_bytes, &a[0], &err));
  OCL_CHECK(err, buffer_b = cl::Buffer(context, CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR , size_b_in_bytes, &b[0], &err));



  // set kernel arguments
  // test_run_index++;
  // std::cout << std::endl;
  // std::cout << "RUN "<< test_run_index << std::endl;
  // std::cout << "Setting kernel arguments...  tA " << tA << "  tB " << tB << "  incC " << incC << std::endl;
  OCL_CHECK(err, err = kernel_ut.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_ut.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_ut.setArg(2, (long int)size));


  //-----------------------------
  // Copy input data to device global memory
  std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_a, buffer_b}, 0 /*0 means from host*/, NULL, &write_event));
  // set_callback(write_events[0], "ooo_queue");
  // q.finish();
  //
  //-----------------------------
  // printf("Enqueueing NDRange kernel.\n");
  // This event needs to wait for the write buffer operations to complete
  // before executing. We are sending the write_events into its wait list to
  // ensure that the order of operations is correct.
  // Launch the Kernel
  // std::vector<cl::Event> waitList;
  // waitList.push_back(write_events[0]);
  // OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_ut, 0, 1, 1, &waitList, &kernel_events[0]));
  // set_callback(kernel_events[0], "ooo_queue");


  // Profiling Objects
  cl_ulong start= 0;
  cl_ulong end = 0;
  double diff_prof = 0.0f;
  std::cout<<"Launching Task "<<std::endl;

  OCL_CHECK(err, err = q.enqueueTask(kernel_ut, NULL, &event));
  clWaitForEvents(1, (const cl_event*) &event);
  // q.finish();
  event.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
  event.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
  diff_prof = end-start;
  std::cout<<"Profiling kernel_relu: "<<(diff_prof/1000000)<<"ms"<<std::endl;




  //-----------------------------
  // Copy Result from Device Global Memory to Host Local Memory
  std::cout << "Getting Results (Device to Host)..." << std::endl;
  // std::vector<cl::Event> eventList;
  // eventList.push_back(kernel_events[0]);
  // This operation only needs to wait for the kernel call.
  // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, CL_MIGRATE_MEM_OBJECT_HOST, &eventList, & <));
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, CL_MIGRATE_MEM_OBJECT_HOST, NULL, &result_ready));
  result_ready.wait();

  OCL_CHECK(err, err = q.finish());

  // printf("B: ");
  // for (int i = 0; i < size_ab; i++) {
  //       printf("%f " , b[i]);
  //     }
  // printf("\n");


  //Verification

  //CPU function
  for(int i = 0; i<size_ab;i++){
    if(A_CPU[i]<0.0)B_CPU[i] = 0;
    else{B_CPU[i]=A_CPU[i];}
  }

  // printf("B_CPU: ");
  // for (int i = 0; i < size_ab; i++) {
  //       printf("%f " , B_CPU[i]);
  //     }
  // printf("\n");

  int error = 0;
  //Compare results
  for(int i = 0; i<size_ab;i++){
    if(B_CPU[i] != b[i]) error++;
  }

  if(error == 0){printf("TEST PASSED ... \n");}
  else{printf("Error ---> %d \n", error);}

  //-----------------------------
  std::cout << "" << std::endl;
  std::cout << "All done" << std::endl;
  std::cout << "quit now" << std::endl;

  // exit
  return 0;
}
