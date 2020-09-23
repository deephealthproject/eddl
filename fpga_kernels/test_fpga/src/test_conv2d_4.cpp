#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "xcl2.hpp"

using std::vector;

// CL
cl::Buffer buf;
cl::Context context;
cl::CommandQueue q;
cl::Program program;


#define W    256 //256
#define H    256 //256
#define C    4  //I
#define COUT 4  //O
#define KW   3
#define KH   3

// buffers
float data_in[  W   * H  * C       ]  __attribute__ ((__aligned__(16)));
float kernel [ KW   * KH * C * COUT]  __attribute__ ((__aligned__(16)));
float bias   [ COUT                ]  __attribute__ ((__aligned__(16)));
float out    [  W   * H  * COUT    ]  __attribute__ ((__aligned__(16)));
float out_cpu[  W   * H  * COUT    ]  __attribute__ ((__aligned__(16)));

void cpu_conv2d() {

  int size_out = W * H * COUT;
  for (int i=0; i<size_out; i++) out_cpu[i] = 0.f;

  for (int c=0; c<C; c++) {
    for (int cout=0; cout<COUT; cout++) {
      for (int h=0; h<H; h++) {
        for (int w=0; w<W; w++) {
          for (int kh=0; kh<KH; kh++) {
	    for (int kw=0; kw<KW; kw++) {
	      int data_h = (h-1)+kh;
	      int data_w = (w-1)+kw;
	      int padding = (data_h == -1) | (data_w == -1) | (data_w == W) | (data_h == H);
	      int addr_k = (c * COUT * KW * KH) + (cout * KW * KH) + (kh * KW) + kw;
              int addr_p = (data_h * W * C) + (data_w * C) + c;
	      int addr_o = (h * W * COUT) + (w * COUT) + cout;
	      if (!padding) out_cpu[addr_o] += data_in[addr_p] * kernel[addr_k];
	    }
	  }
	}
      }
    }
  }

  // añadimos bias
  for (int cout=0; cout<COUT; cout++) {
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
        int addr_o = (h * W * COUT) + (w * COUT) + cout;
        out_cpu[addr_o] += bias[cout];
      }
    }
  }

  // aplicamos relu
/*  for (int cout=0; cout<COUT; cout++) {
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
        int addr_o = (h * W * COUT) + (w * COUT) + cout;
        if (out_cpu[addr_o] < 0.f) out_cpu[addr_o] = 0.f;
      }
    }
  }*/
}

void cpu_print_data_in() {
  printf("data in:\n");
  for (int c=0; c<C; c++) {
    printf(" channel %d:\n", c);
    printf("   ");
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
	int addr_p = (h * W * C) + (w * C) + c;
        printf("%6.2f ", data_in[addr_p]);
      }
      printf("\n");
      printf("   ");
    }
    printf("\n");
  }
}

void cpu_print_kernels() {
  printf("kernels:\n");
  for (int c=0; c<C; c++) {
    for (int cout=0; cout<COUT; cout++) {
      printf("kernel c=%d cout %d:\n", c, cout);
      for (int kh=0; kh<KH; kh++) {
        for (int kw=0; kw<KW; kw++) {
	  int addr_k = (c * COUT * KW * KH) + (cout * KW * KH) + (kh * KW) + kw;
	  printf("%6.2f ", kernel[addr_k]);
	}
	printf("\n");
      }
    }
  }
}

void cpu_print_bias() {
  printf("bias:\n");
  for (int cout=0; cout<COUT; cout++) {
    printf("%6.2f ", bias[cout]);
  }
  printf("\n");
}

void cpu_print_out() {
  printf("output: cpu (fpga)\n");
  for (int cout=0; cout<COUT; cout++) {
    printf("channel %d:\n", cout);
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
        int addr_o = (h * W * COUT) + (w * COUT) + cout;
        printf(" %10.6f (%10.6f) (diff %10.6f) | ", out_cpu[addr_o], out[addr_o], out_cpu[addr_o]-out[addr_o]);
      }
      printf("\n");
    }
  }
}

void check_result() {

  int error = 0;
  for (int cout=0; cout<COUT; cout++) {
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
        int addr_o = (h * W * COUT) + (w * COUT) + cout;
        if (fabs(out_cpu[addr_o] - out[addr_o]) > 0.001) {
          printf("Results mismatch at cout %d h %d w %d: %6.4f %6.4f (diff %6.4f)\n", cout, h, w, out_cpu[addr_o], out[addr_o], fabs(out_cpu[addr_o]-out[addr_o]));
          error = 1;
	  return;
	}
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

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  printf("Test CONV: [WxHxC] = [%dx%dx%d] -> [WxHxC] = [%dx%dx%d] (kernel [%dx%d], stride [1x1], padding [1x1])\n", W, H, C, W, H, COUT, KW, KH);

  std::string binaryFile = argv[1];
  cl_int err;
  cl::Kernel kernel_conv2d_2;

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

  OCL_CHECK(err, kernel_conv2d_2 = cl::Kernel(program,"k_conv2d_4", &err));
  std::cout << "Kernel sucessfully created" << std::endl ;

  size_t size_data_in_bytes = W*H*C*sizeof(float);
  size_t size_output_in_bytes = W*H*COUT * sizeof(float);
  size_t size_kernel_in_bytes = KW * KH * C * COUT * sizeof(float);
  size_t size_bias_in_bytes = COUT * sizeof(float);
  // Allocate memory on the host and fill with random data.

  //-----------------------------
  // fill data vector with random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::cout << "Filling buffer with useful data" << std::endl ;
  int addr = 0;
  for (int h=0; h<H; h++) {
    for (int w=0; w<W; w++) {
      for (int c=0; c<C; c++) {
	       float value = (c*W*H) + (float)(h*W)+w; //c+1; // (float)((c * 25) + (h * W) + w);
         data_in[addr] = dist(gen); //value;
	       addr++;
      }
    }
  }

  std::cout << "Filling kernel buffer with useful data" << std::endl;
  int kernel_id = 1;
  for (int c=0; c<C; c++) {
    for (int cout=0; cout<COUT; cout++) {
      for (int kh=0; kh<KH; kh++) {
	       for (int kw=0; kw<KW; kw++) {
          float value = (float)kernel_id;
          int addr_k = (c * COUT * KW * KH) + (cout * KW * KH) + (kh * KW) + kw;
	         kernel[addr_k] = dist(gen);
        }
      }
      kernel_id++;
    }
  }

  std::cout << "Filling bias buffer with useful data" << std::endl;
  for (int cout=0; cout<COUT; cout++) bias[cout] = cout; //dist(gen);

  //-----------------------------
  // THIS PAIR OF EVENTS WILL BE USED TO TRACK WHEN A KERNEL IS FINISHED WITH
  // THE INPUT BUFFERS. ONCE THE KERNEL IS FINISHED PROCESSING THE DATA, A NEW
  // SET OF ELEMENTS WILL BE WRITTEN INTO THE BUFFER.
  vector<cl::Event> kernel_events(1);
  vector<cl::Event> read_events(1);
  vector<cl::Event> write_events(1);
  cl::Buffer buffer_a;
  cl::Buffer buffer_b;
  cl::Buffer buffer_k;
  cl::Buffer buffer_bias;

  //-----------------------------
  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::cout << "Creating Buffers..." << std::endl;

  OCL_CHECK(err, buffer_a = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_data_in_bytes, &data_in, &err));
  OCL_CHECK(err, buffer_b = cl::Buffer(context, CL_MEM_WRITE_ONLY  | CL_MEM_USE_HOST_PTR , size_output_in_bytes, &out, &err));
  OCL_CHECK(err, buffer_k = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_kernel_in_bytes, &kernel, &err));
  OCL_CHECK(err, buffer_bias = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_bias_in_bytes, &bias, &err));

  // set kernel arguments
  int arg = 0;
  //OCL_CHECK(err, err = kernel_conv2d_2.setArg(arg++, W));
  //OCL_CHECK(err, err = kernel_conv2d_2.setArg(arg++, H));
  OCL_CHECK(err, err = kernel_conv2d_2.setArg(arg++, buffer_a));
  OCL_CHECK(err, err = kernel_conv2d_2.setArg(arg++, buffer_k));
  OCL_CHECK(err, err = kernel_conv2d_2.setArg(arg++, buffer_bias));
  OCL_CHECK(err, err = kernel_conv2d_2.setArg(arg++, buffer_b));

  //-----------------------------
  // Copy input data to device global memory
  std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_a}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");

  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_k}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");

  //-----------------------------
  printf("Enqueueing NDRange kernel.\n");
  // This event needs to wait for the write buffer operations to complete
  // before executing. We are sending the write_events into its wait list to
  // ensure that the order of operations is correct.
  // Launch the Kernel
  std::vector<cl::Event> waitList;
  waitList.push_back(write_events[0]);
  OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_conv2d_2, 0, 1, 1, &waitList, &kernel_events[0]));
  set_callback(kernel_events[0], "ooo_queue");



  std::cout << "Getting Results (Device to Host)..." << std::endl;
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


  std::cout << "computing conv in CPU..." << std::endl;

 // cpu_print_data_in();
  // cpu_print_kernels();
 // cpu_print_bias();
  // cpu_conv2d();
 // cpu_print_out();

  // check_result();

  //-----------------------------
  std::cout << "" << std::endl;
  std::cout << "All done" << std::endl;
  std::cout << "quit now" << std::endl;

  // exit
  return 0;
}
