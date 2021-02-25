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


//
// test_conv2D_8x8. Test for the conv2D_8x8 kernel
//
// The test is multi-kernel and multi-frame
//
// Constants:
//
//  - CPI = 16
//  - CPO = 16
//  - KW = 3
//  - KH = 3
//  - PW = 1
//  - PH = 1
//  - SW = 1
//  - SH = 1
//
//  Arguments:
//
//  - W
//  - H
//  - I
//  - O
//
//  Data formats:
//
//  - kernel   : GO x GI x CPO x CPI x KH x KW
//  - bias     : O
//  - data_in  : I x H x W
//  - data_out : O x H x W
//
//  GO = ceil(O / CPO), GI = ceil(I / CPI)
//
// The kernel must have at least 8 I channels and 8 O channels, filled with zeroes if needed

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <ap_fixed.h>
#include <ap_int.h>
#include <sys/time.h>
#include "CL/cl_ext_xilinx.h"
#include "xcl2.hpp"

using std::vector;

// data type
//#define data_type float
//#define data_type ap_fixed<8,4,AP_TRN,AP_WRAP>
#define data_type ap_int<8>

#define MAX_CONVS        8  // Maximum number of convolutional layers
#define MAX_KERNELS      4  // Maximum number of kernels implemented
#define WMAX           256  // Maximum image width
#define HMAX           256  // Maximum image height
#define CPI              4  // Basic kernel number of input channels
#define CPO              4  // Basic kernel number of output channels
#define KW               3  // Convolutional kernel width
#define KH               3  // Convolutional kernel height
#define RELU             0  // Flag for ReLu activation. Active at high level
#define MAX_WORK_ITEMS 512  // Maximum number of work items to process

pthread_t th_process_kernel[MAX_KERNELS]; // Threads to control the FPGA kernels
struct args {                             // Arguments for the threads
  int kernel;                             //
};                                        //
pthread_mutex_t mutex_work_item;          // Mutex for the threads to access pending work

// Global variables
int CONVS;                       // Number of convolutional layers
int KERNELS;                     // Number of FPGA kernels to use
int F;                           // Number of frames of the data
int W;                           // Width of the data
int H;                           // Height of the data
int I;                           // Number of input channels
int O;                           // Number of output channels
int I_kernel;                    // Number of input channels corrected for the conv kernel
int O_kernel;                    // Number of output channels corrected for the conv kernel
int GI;                          // Number of input groups of channels
int GO;                          // Number of output groups of channels

// buffers
data_type *data_in;               // Input data buffer (format I x W x H)
data_type *out[MAX_CONVS];        // Output data buffer (format O x W x H)
data_type *kernel[MAX_CONVS];     // Conv kernel buffers (format GO x GI x CPO x CPI x KW x KH)
data_type *bias[MAX_CONVS];       // Conv bias buffers (format O)
data_type *out_cpu[MAX_CONVS];    // Output data buffer for cpu (format O x W x H)

// OpenCL variables
cl::Context context;                          // Context
cl::CommandQueue q;                           // Command queue
cl::Program program;                          // Program
std::string binaryFile;                       // Binary file
cl::Kernel kernel_conv2d[MAX_KERNELS];        // FPGA kernels
vector<cl::Event> kernel_events(MAX_KERNELS); // Kernel events (completion)
vector<cl::Event> read_events(1);             // Read events
vector<cl::Event> write_events(3);            // Write events
cl::Buffer *buffer_i;                         // input buffer
cl::Buffer *buffer_o[MAX_CONVS];              // output buffers
cl::Buffer *buffer_k[MAX_CONVS];              // Conv kernel buffers
cl::Buffer *buffer_bias[MAX_CONVS];           // Conv bias buffers
// DDR assignment
cl_mem_ext_ptr_t data_in_ddr;                 // input data buffer
cl_mem_ext_ptr_t out_ddr[MAX_CONVS];          // output data buffers
cl_mem_ext_ptr_t kernel_ddr[MAX_CONVS];       // Conv kernel buffers
cl_mem_ext_ptr_t bias_ddr[MAX_CONVS];         // Conv bias buffers

// Work item
struct work_item_st {
  int F;                        // Frame
  int CONV;                     // Convolution layer
  int H;                        // Height
  int W;                        // Width
  int rows;                     // Number of rows of the frame
  cl::Buffer i;                 // input data buffer
  cl::Buffer o;                 // output data buffer
  cl::Buffer k;                 // conv kernel buffer
  cl::Buffer bias;              // conv bias buffer
  int I;                        // Number of input channels
  int O;                        // Number of output channels
  int global_offset;            // Offset to the frame
  int enable_upper_padding;     // Enable upper padding to the frame
  int enable_lower_padding;     // Enable lower padding to the frame
  int O_ITER;                   // Number of output iterations
  int I_ITER;                   // Number of input iterations
  int enable_relu;              // Enable ReLU activation
  int dep0_in;                  // Input dependency (with another work item)
  int dep1_in;                  // Input dependency (with another work item)
  int valid;                    // Valid entry
  int in_process = 0;           // Work item being in process
};

work_item_st wi[MAX_WORK_ITEMS];  // Work items
int num_work_items = 0;           // Number of work items in the list
int remaining_work_items = 0;     // Remaining work items to be processed

// -------------------------------------------------------------------------------------------------
// Functions
//

// Allocate_buffers. Allocates in CPU memory all the needed buffers
void allocate_buffers() {
  printf("voy a ubicar\n");
  posix_memalign((void **)&data_in, 4096, I * W * H * sizeof(data_type));
  for (int conv=0; conv<CONVS; conv++) {
    posix_memalign((void **)&kernel[conv], 4096, I_kernel * O_kernel * KW * KH * sizeof(data_type));
    posix_memalign((void **)&bias[conv], 4096, O * sizeof(data_type));
    posix_memalign((void **)&out[conv], 4096, O * W * H * sizeof(data_type));
    posix_memalign((void **)&out_cpu[conv], 4096, O * W * H * sizeof(data_type));
  }
  printf("fin ubicar\n");
}

// deallocate_buffers. Deallocates all CPU buffers
void deallocate_buffers() {
  free(data_in);
  for (int conv=0; conv<CONVS; conv++) {
    free(kernel[conv]);
    free(bias[conv]);
    free(out[conv]);
    free(out_cpu[conv]);
  }
}

// parse_arguments. Parses the program arguments
void parse_arguments(int argc, char **argv) {
  if (argc != 9) {
    printf("Syntax:\n%s <XCLBIN File> <CONVS> <F> <KERNELS> <W> <H> <I> <O>\n", argv[0]);
    printf("  <XCLBIN File> : File with the xclbin file\n");
    printf("  <CONVS>       : Number of convolutional layers\n");
    printf("  <F>           : Number of frames to divide the channels into\n");
    printf("  <KERNELS>     : Number of FPGA kernels to use\n");
    printf("  <W>           : Width of input images\n");
    printf("  <H>           : Height of input images\n");
    printf("  <I>           : Number of input channels\n");
    printf("  <O>           : Number of output channels\n");
    exit(1);
  }

  // Get the arguments
  binaryFile = argv[1];
  CONVS = atoi(argv[2]);
  F = atoi(argv[3]);
  KERNELS = atoi(argv[4]);
  W = atoi(argv[5]);
  H = atoi(argv[6]);
  I = atoi(argv[7]);
  O = atoi(argv[8]);

  // check arguments and compute others
  if (KERNELS > MAX_KERNELS) {printf("Error, too many kernels\n"); exit(1);}
  if (I < CPI) {
    I_kernel = CPI;
  } else {
    I_kernel = I;
    if ((I % CPI) != 0) {printf("Error, I must me multiple of %d or lower than %d\n", CPI, CPI); exit(1);}
  }

  if (O < CPO) {
    O_kernel = CPO;
  } else {
    O_kernel = O;
    if ((O % CPO) != 0) {printf("Error, O must be multiple of %d or lower than %d\n", CPO, CPO); exit(1);}
  }
  GI = I_kernel / CPI;
  GO = O_kernel / CPO;
}

// printing function. Prints output data generated by the cpu and by the one obtained from the FPGA
void cpu_print_out() {
  printf("output: cpu (fpga)\n");
  for (int cout=0; cout<O; cout++) {
    printf("channel %d:\n", cout);
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
	      // data_out pixel position
        int addr_o = (cout * W * H) + (h * W) + w;
        printf(" %10.6f (%10.6f) (diff %10.6f) | ", float(out_cpu[CONVS-1][addr_o]), float(out[CONVS-1][addr_o]), float(out_cpu[CONVS-1][addr_o]-out[CONVS-1][addr_o]));
      }
      printf("\n");
    }
  }
}

// cpu_conv2d. Performs the convolutions on the cpu
void cpu_conv2d() {

  for (int conv=0; conv<CONVS; conv++) {
    int size_out = O * W * H;
    for (int i=0; i<size_out; i++) out_cpu[conv][i] = 0.f;

    for (int c=0; c<I; c++) {
      for (int cout=0; cout<O; cout++) {
        for (int h=0; h<H; h++) {
          for (int w=0; w<W; w++) {
            for (int kh=0; kh<KH; kh++) {
	            for (int kw=0; kw<KW; kw++) {
	              int data_h = (h-1)+kh;
	              int data_w = (w-1)+kw;
	              int padding = (data_h == -1) | (data_w == -1) | (data_w == W) | (data_h == H);
	              // kernel position
                int gi = c / CPI;
                int ki = c % CPI;
                int go = cout / CPO;
                int ko = cout % CPO;
                int addr_k = (go * GI * CPO * CPI * KH * KW) +
                             (gi * CPO * CPI * KH * KW) +
                             (ko * CPI * KH * KW) +
                             (ki * KH * KW) +
                             (kh * KW) +
                             kw;
                //int addr_k = (cout * I_kernel * KW * KH) + (c * KW * KH) + (kw * KH) + kh;
	              // data_in pixel position
                int addr_p = (c * W * H) + (data_h * W) + data_w;
	              // data_out pixel position
                int addr_o = (cout * W * H) + (h * W) + w;
	              // operation
                data_type din = (conv==0)?data_in[addr_p]:(out_cpu[conv-1])[addr_p];
	              if (!padding) (out_cpu[conv])[addr_o] += din * (kernel[conv])[addr_k];
	            }
	          }
	        }
        }
      }
    }

    // añadimos bias
    for (int cout=0; cout<O; cout++) {
      for (int h=0; h<H; h++) {
        for (int w=0; w<W; w++) {
	        // data_out pixel position
          int addr_o = (cout * W * H) + (h * W) + w;
	        // bias operation
          out_cpu[conv][addr_o] += bias[conv][cout];
        }
      }
    }

    //añadimos relu
    if(RELU){
      for (int cout=0; cout<O; cout++) {
        for (int h=0; h<H; h++) {
          for (int w=0; w<W; w++) {
            int addr_o = (h * W * O) + (w * O) + cout;
            if (out_cpu[conv][addr_o] < 0.f) out_cpu[conv][addr_o] = 0.f;
          }
        }
      }
    }
  }
}

// printing function. Prints the input data
void cpu_print_data_in() {
  printf("data in:\n");
  for (int c=0; c<I; c++) {
    printf(" channel %d:\n", c);
    printf("   ");
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
      	// data_in pixel position
	      int addr_p = (c * W * H) + (h * W) + w;
        printf("%6.2f ", float(data_in[addr_p]));
      }
      printf("\n");
      printf("   ");
    }
    printf("\n");
  }
}

// printing function. Prints the conv kernels for a specific convolution layer
void cpu_print_kernels(int conv) {
  printf("kernels (conv=%d):\n", conv);
  for (int cout=0; cout<O_kernel; cout++) {
    for (int c=0; c<I_kernel; c++) {
      if ((cout < O) && (c<I)) {
        printf("kernel c=%d cout %d:\n", c, cout);
        for (int kh=0; kh<KH; kh++) {
          for (int kw=0; kw<KW; kw++) {
             // kernel position
            int gi = c / CPI;
            int ki = c % CPI;
            int go = cout / CPO;
            int ko = cout % CPO;
            int addr_k = (go * GI * CPO * CPI * KH * KW) +
                         (gi * CPO * CPI * KH * KW) +
                         (ko * CPI * KH * KW) +
                         (ki * KH * KW) +
                         (kh * KW) +
                         kw;
            //int addr_k = (cout * I_kernel * KW * KH) + (c * KW * KH) + (kh * KW) + kw;
            printf("%6.2f ", float(kernel[conv][addr_k]));
      	  }
	        printf("\n");
        }
      }
    }
  }
}

// printing function. Prints the conv bias for a specific convolution layer
void cpu_print_bias(int conv) {
  printf("bias(conv=%d):\n", conv);
  for (int cout=0; cout<O; cout++) {
    printf("%6.2f ", float(bias[conv][cout]));
  }
  printf("\n");
}

// check_result function. Checks output produced by the CPU and by the FPGA
void check_result() {

  int error = 0;
  for (int cout=0; cout<O; cout++) {
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
      	// data_out pixel position
        int addr_o = (cout * W * H) + (h * W) + w;
        if (fabs(float(out_cpu[CONVS-1][addr_o]) - float(out[CONVS-1][addr_o])) > 0.001) {
          printf("Results mismatch at cout %d h %d w %d: %6.4f %6.4f (diff %6.4f)\n", cout, h, w, float(out_cpu[CONVS-1][addr_o]), float(out[CONVS-1][addr_o]), fabs(float(out_cpu[CONVS-1][addr_o]-out[CONVS-1][addr_o])));
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

// OpenCL event callback function printing message
void event_cb(cl_event event1, cl_int cmd_status, void *data) {
  cl_int err;
  cl_command_type command;
  cl::Event event(event1, true);
  OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_TYPE, &command));
  cl_int status;
  OCL_CHECK(err, err = event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status));
  const char *command_str;
  const char *status_str;
  switch (command) {
    case CL_COMMAND_READ_BUFFER:          command_str = "buffer read"; break;
    case CL_COMMAND_WRITE_BUFFER:         command_str = "buffer write"; break;
    case CL_COMMAND_NDRANGE_KERNEL:       command_str = "kernel"; break;
    case CL_COMMAND_MAP_BUFFER:           command_str = "kernel"; break;
    case CL_COMMAND_COPY_BUFFER:          command_str = "kernel"; break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:  command_str = "buffer migrate"; break;
    default:                              command_str = "unknown";
  }
  switch (status) {
    case CL_QUEUED:       status_str = "Queued"; break;
    case CL_SUBMITTED:    status_str = "Submitted"; break;
    case CL_RUNNING:      status_str = "Executing"; break;
    case CL_COMPLETE:     status_str = "Completed"; break;
  }
  printf("[%s]: %s %s\n", reinterpret_cast<char *>(data), status_str, command_str);
  fflush(stdout);
}

// Sets the callback for a particular event
void set_callback(cl::Event event, const char *queue_name) {
  cl_int err;
  OCL_CHECK(err, err = event.setCallback(CL_COMPLETE, event_cb, (void *)queue_name));
}

// function to retrieve a work item index, based on the convolution layer and the frame
// This function is to set the dependencies between frames
int fn_get_work_item_index(int conv, int f) {
  for (int i=0; i<num_work_items; i++) {
    if (wi[i].valid && (wi[i].CONV == conv) && (wi[i].F == f)) return i;
  }
  printf("Error, work item index not found for conv %d f %d\n", conv, f);
  exit(1);
}

// function to get a work item to process. It is protected with a mutex
int fn_get_work_item(work_item_st *witem, int *wi_index, int *remaining) {
  // search for a work item with no dependencies
  pthread_mutex_lock(&mutex_work_item);
  for (int i=0; i<num_work_items; i++) {
    if (wi[i].valid && (wi[i].dep0_in == -1) && (wi[i].dep1_in == -1) && (!wi[i].in_process)) {
      *witem = wi[i];
      *wi_index = i;
      wi[i].in_process = 1;
      remaining_work_items--;
      *remaining = remaining_work_items;
      pthread_mutex_unlock(&mutex_work_item);
      return 1;
    }
  }
  *remaining = remaining_work_items;
  pthread_mutex_unlock(&mutex_work_item);
  return 0;
}

// function to set a work item as processed and clear possible dependencies. It is protected by a mutex
void fn_processed_work_item(int windex) {
  pthread_mutex_lock(&mutex_work_item);
  wi[windex].valid = 0;
  for (int i=0; i<num_work_items; i++) {
    if (wi[i].valid && (wi[i].dep0_in == windex)) wi[i].dep0_in = -1;
    if (wi[i].valid && (wi[i].dep1_in == windex)) wi[i].dep1_in = -1;
  }
  pthread_mutex_unlock(&mutex_work_item);
}

// process kernel function
void *fn_process_kernel(void *a) {

  int k = ((struct args*)a)->kernel;
  printf("kernel with id %d launched\n", k);

  int err;
  work_item_st wi;
  int wi_index;
  int remaining;
  do {
    int dummy = fn_get_work_item(&wi, &wi_index, &remaining);
    if (dummy) {
      printf("work item: CONV %d F %d\n", wi.CONV, wi.F);

      printf("launching kernel %d\n", k);

      // set kernel arguments
      int arg = 0;
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.i));
      //OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.i));
      //OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.i));
      //OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.i));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.H));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.W));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.rows));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.I));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.O));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.I_ITER));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.O_ITER));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.enable_relu));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.k));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.bias));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.o));
      //OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.o));
      //OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.o));
      //OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.o));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.global_offset));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.enable_upper_padding));
      OCL_CHECK(err, err = kernel_conv2d[k].setArg(arg++, wi.enable_lower_padding));

      OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_conv2d[k], 0, 1, 1, NULL, &kernel_events[k]));
        // OCL_CHECK(err, err = q.enqueueTask(kernel_conv2d[k]));
      set_callback(kernel_events[k], "ooo_queue");
      OCL_CHECK(err, err = kernel_events[k].wait());
      // next work item in the work line is ready
      fn_processed_work_item(wi_index);
    }
  } while (remaining != 0);

  printf("exit process kernel %d\n", k);
}

void fn_add_work_item(work_item_st witem) {
  int item = num_work_items;
  if (item == MAX_WORK_ITEMS) {printf("limit of work items reached\n"); exit(1);}
  wi[item] = witem;
  wi[item].in_process = 0;
  printf("work item added index %d: (CONV %d F %d H %d W %d rows %d I %d O %d go %d eup %d elp %d O_ITER %d I_ITER %d erelu %d dep0_in %d dep1_in %d)\n", item, witem.CONV, witem.F, witem.H, witem.W,
                                        witem.rows, witem.I, witem.O, witem.global_offset, witem.enable_upper_padding, witem.enable_lower_padding,
                                        witem.O_ITER, witem.I_ITER,  witem.enable_relu, witem.dep0_in, witem.dep1_in);
  num_work_items++;
  remaining_work_items++;
}



void fn_compute_work_items() {

  // We go conv by conv
  for (int conv=0; conv<CONVS; conv++) {
    for (int f=0; f<F; f++) {
      work_item_st wi;
      wi.F = f;
      wi.CONV = conv;
      wi.H = H;
      wi.W = W;
      wi.rows = H/F;
      wi.i = (conv==0)?*buffer_i:*buffer_o[conv-1];
      wi.I = I;
      wi.o = *buffer_o[conv];
      wi.O = O;
      wi.k = *buffer_k[conv];
      wi.bias = *buffer_bias[conv];
      wi.global_offset = f * (H/F) * W;
      wi.enable_upper_padding = (f==0);
      wi.enable_lower_padding = (f==F-1);
      wi.O_ITER = (O + (CPO-1)) / CPO;
      wi.I_ITER = (I + (CPI-1)) / CPI;
      wi.enable_relu = RELU;
      wi.dep0_in = conv==0?-1:fn_get_work_item_index(conv-1, f);
      wi.dep1_in = conv==0?-1:f==F-1?-1:fn_get_work_item_index(conv-1, f+1);
      wi.valid = 1;
      fn_add_work_item(wi);
    }
  }
}



//---------------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv) {

  parse_arguments(argc, argv);

  printf("Test:\n");
  printf("  Number of Convolutions : %d\n", CONVS);
  printf("  Input shape            : [%d ch x %d rows x %d cols]\n", I, H, W);
  printf("  Ouput shape            : [%d ch x %d rows x %d cols]\n", O, H, W);
  printf("  Kernel size            : %d x %d\n", KW, KH);
  printf("  Stride                 : 1 x 1\n");
  printf("  Padding                : 1 x 1\n");
  printf("  Apply RELU             : %s\n", RELU?"Yes":"No");
  printf("  Number of kernels      : %d\n", KERNELS);
  printf("  Number of frames       : %d\n", F);

  cl_int err;

  std::cout << "Creating Context..." << std::endl;
  auto devices = xcl::get_xil_devices();
  auto device = devices[0];
  OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
  OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);

  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
  std::cout << "Device " << device_name.c_str() << ": program successful!" << std::endl;

  // kernels
  for (int kernel=0; kernel<KERNELS; kernel++) {
    char dummy[50];
    sprintf(dummy, "k_conv2D:{k_conv2D_%d}", kernel+1);
    printf("dummy %s\n", dummy);
    OCL_CHECK(err, kernel_conv2d[kernel] = cl::Kernel(program, dummy, &err));
    std::cout << "Kernel sucessfully created" << std::endl ;
  }

  size_t size_data_in_bytes = W * H * I * sizeof(data_type);
  size_t size_output_in_bytes = W * H * O * sizeof(data_type);
  size_t size_kernel_in_bytes = KW * KH * I_kernel * O_kernel * sizeof(data_type);
  size_t size_bias_in_bytes = O * sizeof(data_type);

  // Allocate memory on the host and fill with random data.
  allocate_buffers();

  //-----------------------------
  // fill data vector with random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::cout << "Filling buffer with useful data" << std::endl;
  int addr = 0;
  for (int i=0; i<I; i++) {
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
          data_type value = data_type(i+1); //data_type((i * W * H) + (h * W) + w); //c+1; // (data_type)((c * 25) + (h * W) + w);
          data_in[addr] = value; //dist(gen); //value;
          addr++;
      }
    }
  }

  for (int conv=0; conv<CONVS; conv++) {
    std::cout << "Filling kernel buffer with useful data" << std::endl;
    int kernel_id = 1;
    for (int i=0; i<I_kernel; i++) {
      for (int o=0; o<O_kernel; o++) {
        for (int kh=0; kh<KH; kh++) {
	        for (int kw=0; kw<KW; kw++) {
            data_type value = (data_type)kernel_id;
            if ((o >= O) || (i >= I)) value = (data_type) 0;
            int gi = i / CPI;
            int ki = i % CPI;
            int go = o / CPO;
            int ko = o % CPO;
            int addr_k = (go * GI * CPO * CPI * KH * KW) +
                         (gi * CPO * CPI * KH * KW) +
                         (ko * CPI * KH * KW) +
                         (ki * KH * KW) +
                         (kh * KW) +
                         kw;
            //int addr_k = (o * I_kernel * KW * KH) + (i * KW * KH) + (kh * KW) + kw;
            if ((i<I) && (o<O)) kernel[conv][addr_k] = dist(gen);//value; //dist(gen);
            else kernel[conv][addr_k] = 0;
          }
        }
        if ((o < O) && (i < I)) kernel_id++;
      }
    }

    std::cout << "Filling bias buffer with useful data" << std::endl;
    for (int cout=0; cout<O; cout++) bias[conv][cout] = dist(gen);
  }

  //-----------------------------
  // THIS PAIR OF EVENTS WILL BE USED TO TRACK WHEN A KERNEL IS FINISHED WITH
  // THE INPUT BUFFERS. ONCE THE KERNEL IS FINISHED PROCESSING THE DATA, A NEW
  // SET OF ELEMENTS WILL BE WRITTEN INTO THE BUFFER.

  //DATA IN
  data_in_ddr.flags  =  0 | XCL_MEM_TOPOLOGY;
  data_in_ddr.obj = data_in; //puntero a los datos
  data_in_ddr.param = 0; //parametro reservado para actualizaciones futuras
  for (int conv=0; conv<CONVS; conv++) {
    //OUT
    out_ddr[conv].flags  = 0 | XCL_MEM_TOPOLOGY;
    out_ddr[conv].obj = out[conv];
    out_ddr[conv].param = 0;
    //KERNEL
    kernel_ddr[conv].flags  = 0 | XCL_MEM_TOPOLOGY;
    kernel_ddr[conv].obj = kernel[conv];
    kernel_ddr[conv].param = 0;
    //BIAS
    bias_ddr[conv].flags  = 0 | XCL_MEM_TOPOLOGY;
    bias_ddr[conv].obj = bias[conv];
    bias_ddr[conv].param = 0;
  }

  //-----------------------------
  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::cout << "Creating Buffers..." << std::endl;


 //se cre el buffer en relación al puntero asignado al banco de memoria de la tarjeta y se añade el flag CL_MEM_EXT_PTR_XILINX
  OCL_CHECK(err, buffer_i = new cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_data_in_bytes, &data_in_ddr, &err));

  for (int conv=0; conv<CONVS; conv++) {
    printf("conv %d\n", conv);
    OCL_CHECK(err, buffer_o[conv] = new cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY  | CL_MEM_USE_HOST_PTR , size_output_in_bytes, &out_ddr[conv], &err));
    OCL_CHECK(err, buffer_k[conv] = new cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_kernel_in_bytes, &kernel_ddr[conv], &err));
    OCL_CHECK(err, buffer_bias[conv] = new cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_bias_in_bytes, &bias_ddr[conv], &err));
  }


  //-----------------------------
  // Copy input data to device global memory
  // std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {*buffer_i}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");
  OCL_CHECK(err, err = write_events[0].wait());

  for (int conv=0; conv<CONVS; conv++) {
    printf("hola conv %d\n", conv);
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {*buffer_k[conv]}, 0 /*0 means from host*/, NULL, &write_events[0]));
    set_callback(write_events[0], "ooo_queue");
    OCL_CHECK(err, err = write_events[0].wait());

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {*buffer_bias[conv]}, 0 /*0 means from host*/, NULL, &write_events[0]));
    set_callback(write_events[0], "ooo_queue");
    OCL_CHECK(err, err = write_events[0].wait());
  }


  std::cout << "Running kernels..." << std::endl;

  // timint stats
  unsigned long long prof_time;
  struct timeval prof_t1;
  gettimeofday(&prof_t1, NULL);

  // we compute the list of work items
  fn_compute_work_items();

  // we create a thread for each kernel
  for (int kernel=0; kernel<KERNELS; kernel++) {
    printf("launching kernel %d\n", kernel);
    args *a = (struct args *)malloc(sizeof(args));
    a->kernel = kernel;
    int ret = pthread_create(&th_process_kernel[kernel], NULL, fn_process_kernel, (void *)a);
    if (ret) {fprintf(stderr, "Error creating thread process with code: %d\n", ret); exit(1);}
  }

  // and wait them to finish
  for (int kernel=0; kernel<KERNELS; kernel++) {
    pthread_join(th_process_kernel[kernel], NULL);
  }

  // timing
  struct timeval prof_t2;
  gettimeofday(&prof_t2, NULL);
  prof_time = ((prof_t2.tv_sec - prof_t1.tv_sec) * 1000000) + (prof_t2.tv_usec - prof_t1.tv_usec);
  printf("Timing: %8lld usec\n", prof_time);

  // std::cout << "Getting Results (Device to Host)..." << std::endl;
  std::vector<cl::Event> eventList;
  eventList.push_back(kernel_events[0]);
  // This operation only needs to wait for the kernel call.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({*buffer_o[CONVS-1]}, CL_MIGRATE_MEM_OBJECT_HOST, NULL, &read_events[0]));
  set_callback(read_events[0], "ooo_queue");
  OCL_CHECK(err, err = read_events[0].wait());

  // Wait for all of the OpenCL operations to complete
  std::cout << "Waiting..." << std::endl;
  OCL_CHECK(err, err = q.flush());
  OCL_CHECK(err, err = q.finish());

  std::cout << "computing conv in CPU..." << std::endl;

 // cpu_print_data_in();
 //  for (int conv=0; conv<CONVS; conv++) {
 //   cpu_print_kernels(conv);
 //   cpu_print_bias(conv);
 //  }
  cpu_conv2d();
  //cpu_print_out();

  check_result();



  //-----------------------------
  std::cout << "" << std::endl;
  std::cout << "All done" << std::endl;
  std::cout << "quit now" << std::endl;

  deallocate_buffers();


  printf("exit!!!! (to avoid segfault)\n");
  exit(1);

  // exit
  return 0;
}
