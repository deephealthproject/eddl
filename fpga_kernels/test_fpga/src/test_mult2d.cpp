#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "xcl2.hpp"

//#define VERBOSE
//#define DEBUG

using std::vector;

//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

// An event callback function that prints the operations performed by the OpenCL
// runtime.
void event_cb(
    cl_event event1,
    cl_int cmd_status
    , void *data
) {
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

//---------------------------------------------------------------------------------------------------------------------
// Sets the callback for a particular event
void set_callback(
    cl::Event event,
    const char *queue_name
) {
  cl_int err;
  OCL_CHECK(err, err = event.setCallback(CL_COMPLETE, event_cb, (void *)queue_name));
}

//---------------------------------------------------------------------------------------------------------------------
void usage (
    char *p_name
) {
  std::cout << "ERROR: unexpected number of parameters" << std::endl;
  std::cout << "Usage: " << p_name << " <XCLBIN File>" << " <Ashape0> <Ashape1> <Bshape0> <Bshape1> <tA> <tB> <incC>" << std::endl;
  exit(EXIT_FAILURE);
}

//---------------------------------------------------------------------------------------------------------------------
void fpga_init(
    cl::Context      &context, 
    cl::CommandQueue &q, 
    cl::Program      &program, 
    cl::Kernel       &kernel_ut,
    const char       *fname, // fpga device binary file name
    const char       *k_name // kernel name
) { 
  cl_int err;
  std::string binaryFile = fname;

  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  std::cout << "Creating Context..." << std::endl;
  // The get_xil_devices will return vector of Xilinx Devices
  auto devices = xcl::get_xil_devices();
  auto device = devices[0];

//  std::vector<float, aligned_allocator<float>> host_memory(elements, 42);
  // Creating Context and Command Queue for selected Device
  OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));

  std::cout << "  setting command queue" << std::endl;
  OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  std::cout << "Allocating and transferring binary file to " << device_name.c_str() << std::endl;

  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);


  std::cout << "Loading program to " << device_name.c_str() << std::endl;
  OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));
  std::cout << "  ... program successful!" << std::endl;

  std::cout << "Creating kernel in program" << std::endl;
  OCL_CHECK(err, kernel_ut = cl::Kernel(program, k_name, &err));
  std::cout << "  ... kernel sucessfully created" << std::endl ;

}

//---------------------------------------------------------------------------------------------------------------------
void create_buffers(
    cl::Context &context,
    cl::Buffer &buffer_a,
    cl::Buffer &buffer_b,
    cl::Buffer &buffer_c,
    vector<float, aligned_allocator<float>> &a,
    vector<float, aligned_allocator<float>> &b,
    vector<float, aligned_allocator<float>> &c,
    size_t size_a_in_bytes,
    size_t size_b_in_bytes,
    size_t size_c_in_bytes

) {
  cl_int err;
  
  //-----------------------------
  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::cout << "Creating Buffers..." << std::endl;

  OCL_CHECK(err, buffer_a = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_a_in_bytes, &a[0], &err));
  OCL_CHECK(err, buffer_b = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_b_in_bytes, &b[0], &err));
  // buffer c will be used for write, and depending on the params can also be read (incremental mmul)
  OCL_CHECK(err, buffer_c = cl::Buffer(context,                     CL_MEM_USE_HOST_PTR , size_c_in_bytes, &c[0], &err));
}

//---------------------------------------------------------------------------------------------------------------------
void fill(
    vector<float, aligned_allocator<float>> &a,
    vector<float, aligned_allocator<float>> &b,
    vector<float, aligned_allocator<float>> &c,
    vector<float, aligned_allocator<float>> &c_local,
    int Ashape0, int Ashape1,
    int Bshape0, int Bshape1,
    int Cshape0, int Cshape1
) {
    // Set/Initialize matrices
  // fill data vectors
  int val = 0;
  std::cout << "Filling matrix A[" << Ashape0 << " , " << Ashape1 << "] with sequential values" << std::endl ;
  for (int i = 0; i < Ashape0; i++) {
    for (int j = 0; j < Ashape1; j++) {
      int ind = i*Ashape1 + j;
      a[ind] = val;
      val += 1;
    }
  }

  std::cout << "Filling matrix B[" << Bshape0 << " , " << Bshape1 << "] to be the Identity Matrix" << std::endl ;
  for (int i = 0; i < Bshape0; i++) {
    for (int j = 0; j < Bshape1; j++) {
      int ind = i*Bshape1 + j;
      b[ind] = (i == j) ? 1 : 0;
    }
  }

  std::cout << "result matrix C will be dimensioned to: C[" << Cshape0 << ", " << Cshape1 << "]" << std::endl;
  std::cout << "Filling matrix C[" << Cshape0 << " , " << Bshape1 << "] with 1s" << std::endl ;
  for (int i = 0; i < Cshape0; i++) {
    for (int j = 0; j < Cshape1; j++) {
      int ind = i*Cshape1 + j;
      c[ind]       = 1;
      c_local[ind] = c[ind];
    }
  }

}

//---------------------------------------------------------------------------------------------------------------------
void run(
    cl::Context      &context,
    cl::CommandQueue &q,
    cl::Kernel       &kernel_ut,
    cl::Buffer       &buffer_a,
    cl::Buffer       &buffer_b,
    cl::Buffer       &buffer_c,
    int Ashape0, int Ashape1,
    int Bshape0, int Bshape1,
    int tA, int tB, int incC
) {
  cl_int err;

  // These events will be used to track when a kernel is finished with
  // the input and output buffers. Once the kernel is finished processing the data,
  // a new set of elements will be written into the output buffer.
  vector<cl::Event> kernel_events(1);
  vector<cl::Event> read_events(1);
  vector<cl::Event> write_events(1);


  //-----------------------------
  // These events will be used to track when a kernel is finished with
  // the input and output buffers. Once the kernel is finished processing the data,
  // a new set of elements will be written into the output buffer.
  //vector<cl::Event> kernel_events(1);
  //vector<cl::Event> read_events(1);
  //vector<cl::Event> write_events(1);
  
  // set kernel arguments
  //test_run_index++;
  std::cout << std::endl;
  //std::cout << "RUN "<< test_run_index << std::endl;
  std::cout << "Setting kernel arguments...  tA " << tA << "  tB " << tB << "  incC " << incC << std::endl;
  OCL_CHECK(err, err = kernel_ut.setArg(0, buffer_a));
  OCL_CHECK(err, err = kernel_ut.setArg(1, buffer_b));
  OCL_CHECK(err, err = kernel_ut.setArg(2, buffer_c));
  OCL_CHECK(err, err = kernel_ut.setArg(3, Ashape0));
  OCL_CHECK(err, err = kernel_ut.setArg(4, Ashape1));
  OCL_CHECK(err, err = kernel_ut.setArg(5, Bshape0));
  OCL_CHECK(err, err = kernel_ut.setArg(6, Bshape1));
  OCL_CHECK(err, err = kernel_ut.setArg(7, tA));
  OCL_CHECK(err, err = kernel_ut.setArg(8, tB));
  OCL_CHECK(err, err = kernel_ut.setArg(9, incC));

  //-----------------------------
  // Copy input data to device global memory
  std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_a, buffer_b, buffer_c}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");

  //-----------------------------
  printf("Enqueueing NDRange kernel.\n");
  // This event needs to wait for the write buffer operations to complete
  // before executing. We are sending the write_events into its wait list to
  // ensure that the order of operations is correct.
  // Launch the Kernel
  std::vector<cl::Event> waitList;
  waitList.push_back(write_events[0]);
  OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_ut, 0, 1, 1, &waitList, &kernel_events[0]));
  set_callback(kernel_events[0], "ooo_queue");

  //-----------------------------
  // Copy Result from Device Global Memory to Host Local Memory
  std::cout << "Getting Results (Device to Host)..." << std::endl;
  std::vector<cl::Event> eventList;
  eventList.push_back(kernel_events[0]);
  // This operation only needs to wait for the kernel call. 
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_c}, CL_MIGRATE_MEM_OBJECT_HOST, &eventList, &read_events[0]));
  set_callback(read_events[0], "ooo_queue");
  OCL_CHECK(err, err = read_events[0].wait());

  std::cout << " Matrix C retrieved from device memory" << std::endl;

  //-----------------------------
  // HEY !!!!
  // It is necessary to release the resources, all of them,
  //  memories, buffers, kernels, programs,...

  // Wait for all of the OpenCL operations to complete
  std::cout << "Waiting for all the operations to complete..." << std::endl;
  OCL_CHECK(err, err = q.flush());
  OCL_CHECK(err, err = q.finish());

  // clear event queues
  kernel_events.clear();
  kernel_events.shrink_to_fit();
  read_events.clear();
  read_events.shrink_to_fit();
  write_events.clear();
  write_events.shrink_to_fit();

}

//---------------------------------------------------------------------------------------------------------------------
void run_cpu(const vector<float, aligned_allocator<float>> &a,
             const vector<float, aligned_allocator<float>> &b,
             //const vector<float, aligned_allocator<float>> &c,
             vector<float, aligned_allocator<float>> &c_local,
             int Ashape0, int Ashape1,
             int Bshape0, int Bshape1,
             int Cshape0, int Cshape1,
             int tA, int tB, int incC
) {
  int *fA_sum, *fA_mult, *fB_sum, *fB_mult;
  int kmax; // common dimension, 
  int i, j, k;

  std::cout << std::endl;
  std::cout << "Performing kernel opeation in CPU" << std::endl;

  if (tA == 0) {
    fA_mult = &i;
    fA_sum  = &k;
    kmax = Ashape1;
  }
  else
  {
    fA_mult = &k;
    fA_sum  = &i;
    kmax = Ashape0;
  }

  if (tB == 0) {
    fB_mult = &k;
    fB_sum  = &j;
  }
  else
  {
    fB_mult = &j;
    fB_sum  = &k;
  }
  
  #ifdef VERBOSE
  std::cout << "c_local" << std::endl;
  for (i = 0; i < Cshape0; i++) {
    for (j = 0; j < Cshape1; j++) {
      int   ind_c = i * Cshape1 + j;
      std::cout << "C[" << i << "][" << j << "] = " << c_local[ind_c] << std::endl;
    }
  }
  #endif

  for (i = 0; i < Cshape0; i++) {
    for (j = 0; j < Cshape1; j++) {
      int   ind_c = i * Cshape1 + j;
      float sum   = 0.0f;

      #ifdef VERBOSE
      std::cout << "C[" << i << "][" << j << "] = ";
      #endif
      for (k = 0; k < kmax; k++) {
        int ind_x = ((*fA_mult) * Ashape1) + (*fA_sum);
        int ind_y = ((*fB_mult) * Bshape1) + (*fB_sum);
        sum += a[ind_x] * b[ind_y];
        #ifdef VERBOSE
        std::cout << "a[" << ind_x << "] * b[" << ind_y << "] ";
        if (k < (kmax -1)) std::cout << "+ ";
        #endif
      }
        #ifdef VERBOSE
        std::cout << " = " << c_local[ind_c] <<  " + " <<  sum  << " = " << (c_local[ind_c] + sum) << std::endl;
        #endif
        c_local[ind_c] = (incC ? c_local[ind_c]:0) + sum;
    }
  }

  #ifdef VERBOSE
  std::cout << "CPU result" << std::endl;
  for (i = 0; i < Cshape0; i++) {
    for (j = 0; j < Cshape1; j++) {
      int   ind_c = i * Cshape1 + j;
      std::cout << "C[" << i << "][" << j << "] = " << c_local[ind_c] << std::endl;
    }
  }
  #endif
}

//---------------------------------------------------------------------------------------------------------------------
// return status of comparison
//     ret 1 - matrices match
//             otherwise return 0
int compare( const vector<float, aligned_allocator<float>> &c,
             const vector<float, aligned_allocator<float>> &c_local,
             size_t size
) {
  int matrices_match = 1;

  for(size_t i = 0; i < size; i++) {
    if (c[i] != c_local[i]) {
      std::cout << "Data mismatch found" << std::endl;
      matrices_match = 0;
      break;
    }
  }
  return matrices_match;
}

//---------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  int Ashape0;
  int Ashape1;
  int Bshape0;
  int Bshape1;
  int Cshape0;
  int Cshape1 ;
  int tA;
  int tB;
  int incC;

  int test_ok      = 1;
  //int test_run_index = 0;

  // CL 
  cl::Context      context;
  cl::CommandQueue q;
  cl::Program      program;
  cl::Kernel       kernel_ut;
  cl::Buffer       buffer_a, buffer_b, buffer_c;

  //---------------------------------------------------------------------------
  if (argc != 9) {
    usage(argv[0]);
  }

  Ashape0 = atoi(argv[2]);
  Ashape1 = atoi(argv[3]);
  Bshape0 = atoi(argv[4]);
  Bshape1 = atoi(argv[5]);
  tA   = atoi(argv[6]); 
  tB   = atoi(argv[7]); 
  incC = atoi(argv[8]); 
 
  // check input configuration
  {
    int ic_err = 0;
    if ((tA == 0) && (tB == 0)) {
      if (Ashape1 != Bshape0) {
        ic_err = 1;
      }
    }
    else if ((tA == 0) && (tB == 1)) {
      if (Ashape1 != Bshape1) {
        ic_err = 1;
      }
    }
    else if ((tA == 1) && (tB == 0)) {
      if (Ashape0 != Bshape0) {
        ic_err = 1;
      }
    }
    else if ((tA == 1) && (tB == 1)) {
      if (Ashape0 != Bshape1) {
        ic_err = 1; 
      }
    }
    else {
      std::cout << "Unexpected configuration" << std::endl;
      ic_err = 1;
    }
  
    if (ic_err != 0) {
      std::cout << "Error matrix dimensions mismatch for requested operation" << std::endl << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  //---------------------------------------------------------------------------
  std::ofstream outfile;
  std::string   outfname = "output.txt";
  outfile.open (outfname.c_str()); // we delete file content by open/close operations
  outfile.close (); //we close the file in case any error happens and the test exits before completion

  //---------------------------------------------------------------------------
  // set matrices dimensions 
  Cshape0 = tA? Ashape1:Ashape0;
  Cshape1 = tB? Bshape0:Bshape1;
  
  size_t size_a = Ashape0 * Ashape1;
  size_t size_b = Bshape0 * Bshape1;
  size_t size_c = Cshape0 * Cshape1;
  size_t size_a_in_bytes = size_a * sizeof(float);
  size_t size_b_in_bytes = size_b * sizeof(float);
  size_t size_c_in_bytes = size_c * sizeof(float);

  // Allocate memory on the host
  vector<float, aligned_allocator<float>> a(size_a, 0);
  vector<float, aligned_allocator<float>> b(size_b, 0);
  vector<float, aligned_allocator<float>> c(size_c, 0);
  vector<float, aligned_allocator<float>> c_local(size_c, 0);
  
  std::cout << "tA " << tA << "  tB " << tB << "  incC " << incC << std::endl;
  std::cout << "A[" << Ashape0 << "x" << Ashape1 << "]   B[" << Bshape0 << "x" << Bshape1 << "]  C[" << Cshape0 << "x" << Cshape1 << "] " << std::endl;
  //---------------
  // fill matrices
  fill(a, b, c, c_local, Ashape0, Ashape1, Bshape0, Bshape1, Cshape0, Cshape1);

  //---------------------------------------------------------------------------
  // Initialize fpga, load binary and kernel
  fpga_init(context, q, program, kernel_ut, argv[1], "k_mult2d");

  // create CL buffers
  create_buffers(context, buffer_a, buffer_b, buffer_c, a, b, c, size_a_in_bytes, size_b_in_bytes, size_c_in_bytes);

  // Run the kernel
  run(context, q, kernel_ut, buffer_a, buffer_b, buffer_c, Ashape0, Ashape1, Bshape0, Bshape1, tA, tB, incC);

  // locally calculate result
  run_cpu(a, b, c_local, Ashape0, Ashape1, Bshape0, Bshape1, Cshape0, Cshape1, tA,tB,incC);

  // compare results
  test_ok = compare (c, c_local, size_c);


  outfile.open(outfname.c_str(), std::ofstream::out | std::ofstream::app);
  if (test_ok != 0) {
    std::cout << "" << std::endl;
    std::cout << "TEST PASSED" << std::endl << std::endl;

    outfile << "" << std::endl;
    outfile << "TEST PASSED" << std::endl << std::endl;
  }
  else {
    std::cout << "" << std::endl;
    std::cout << "ERRORS DETECTED" << std::endl << std::endl;
    std::cout << "TEST KO" << std::endl;

    outfile << "" << std::endl;
    outfile << "ERRORS DETECTED" << std::endl << std::endl;
    outfile << "TEST KO" << std::endl;
  }
  outfile.close();


  //-----------------------------
  // It is necessary to release the resources, all of them,
  a.clear();
  b.clear();
  c.clear();
  c_local.clear();
  a.shrink_to_fit();
  b.shrink_to_fit();
  c.shrink_to_fit();
  c_local.shrink_to_fit();

  //-----------------------------
  std::cout << "" << std::endl;
  std::cout << "All done" << std::endl;
  std::cout << "quit now" << std::endl;

  // exit
  return 0;
}
