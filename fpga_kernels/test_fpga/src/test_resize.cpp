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


#define W    2
#define WOUT 5 //256
#define H    3
#define HOUT 7 //256
#define C    8  //canales totales de entrada
#define COUT 8 //canales totales de salida
#define CPI  8 //nuemro de canales por slice con los que trabajamos
#define CPO	 8
#define GI 	 C/CPI //tamaño del grupo, nº canales entre tamaño slice


// pixel_in
struct pixel_in_t {
  float pixel[CPI]; //pixel de 4 datos
};

// buffers
float data_in[  W   * H   * C    ]  __attribute__ ((__aligned__(16)));
float out    [  WOUT   * HOUT   * COUT ]  __attribute__ ((__aligned__(16)));
pixel_in_t dataSend[W * H * GI] __attribute__ ((__aligned__(16)));




void cpu_print_data_in() {
  printf("data in:\n");
/*     for (int column = 0; column < W; column++) {
      for (int row = 0; row < H; row++) {
        printf("%10.6f ", data_in[column + row * W]);
      }
      printf("\n");
    } */
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
/*   for(int i = 0; i<W * H * C; i++){
	  printf("%6.2f ", data_in[i]);
	  if((i + 1) % W == 0){
		  printf("\n ");
	  }
	  if((i+1) % (H*W) ==0){
		  printf("\n ");
	  }
	  
  } */
  
}


void cpu_print_out() {
  printf("output: fpga\n");
/*     for (int column = 0; column < W; column++) {
      for (int row = 0; row < H; row++) {
        printf("%10.6f ", out[column + row * W]);
      }
      printf("\n");
    } */
	

/*     int k =0;
	int pivote = 0;

	for (int c=0; c<C; c++) {
		if(c < 4){
			pivote = c;
		}else{
		    printf(" K = %d ", k);
			pivote = ((C/2) * HOUT * WOUT) + k;
			k++;
		}
		printf(" channel %d:\n", c);
		printf("   ");
		for (int h=0; h<HOUT; h++) {
		  for (int w=0; w<WOUT; w++) {
			 printf("%6.2f ", out[pivote]);
			pivote+=4;
		  }
		  printf("\n");
		  printf("   ");
		}
		printf("\n");
  } */
  
	for (int c=0; c<C; c++) {
    printf(" channel %d:\n", c);
    printf("   ");
    for (int h=0; h<HOUT; h++) {
      for (int w=0; w<WOUT; w++) {
	int addr_p = (h * WOUT * C) + (w * C) + c;
        printf("%6.2f ", out[addr_p]);
      }
      printf("\n");
      printf("   ");
    }
    printf("\n");
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

  printf("Test flip ");

  std::string binaryFile = argv[1];
  cl_int err;
  cl::Kernel kernel_flip;

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

  OCL_CHECK(err, kernel_flip = cl::Kernel(program,"k_resize", &err));
  std::cout << "Kernel sucessfully created" << std::endl ;

  size_t size_data_in_bytes = W*H*sizeof(float)*C;
  size_t size_output_in_bytes = WOUT*HOUT * sizeof(float)*COUT;


  std::cout << "Filling buffer with useful data" << std::endl ;
  
  int addr = 0;
  
	for (int g=0; g<GI; g++){ //
		for (int h=0; h<H; h++) {
			for (int w=0; w<W; w++) {
				for (int cpi=0; cpi<CPI; cpi++) {
					//float value = (c*W*H) + (float)(h*W)+w; //c+1; // (float)((c 25) + (h W) + w);
					data_in[addr] = addr;//cpi + CPI*g;//dist(gen); //value;-> los datos van seguidos 0 1 2 3 4 5 6 7 8...
					addr++;
				}
			}
		}
	}
	
	addr=0;
	for( int i_iter = 0; i_iter<GI; i_iter++){
      for (int r=0; r<H*W; r++) {
        int addr_d = CPI*i_iter + GI*CPI*r;
		
        dataSend[addr].pixel[0] = data_in[addr_d];
		dataSend[addr].pixel[1] = data_in[addr_d+1];
		dataSend[addr].pixel[2] = data_in[addr_d+2];
		dataSend[addr].pixel[3] = data_in[addr_d+3];
		dataSend[addr].pixel[4] = data_in[addr_d+4];
		dataSend[addr].pixel[5] = data_in[addr_d+5];
		dataSend[addr].pixel[6] = data_in[addr_d+6];
		dataSend[addr].pixel[7] = data_in[addr_d+7];
		
/*  		printf("addr_d = %d \n", addr_d);
        printf("buffer[%d].pixel[0] = %6.2f  ", addr,  dataSend[addr].pixel[0]);
		printf("buffer[%d].pixel[1] = %6.2f  ", addr,  dataSend[addr].pixel[1]);
		printf("buffer[%d].pixel[2] = %6.2f  ", addr, dataSend[addr].pixel[2]);
		printf("buffer[%d].pixel[3] = %6.2f  ", addr, dataSend[addr].pixel[3]);
		printf("\n"); */
		addr++;
	  }
	}

	cpu_print_data_in();

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

  OCL_CHECK(err, buffer_a = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_data_in_bytes, &dataSend, &err));
  OCL_CHECK(err, buffer_b = cl::Buffer(context, CL_MEM_WRITE_ONLY  | CL_MEM_USE_HOST_PTR , size_output_in_bytes, &out, &err));
/*   OCL_CHECK(err, buffer_k = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_kernel_in_bytes, &kernel, &err));
  OCL_CHECK(err, buffer_bias = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_bias_in_bytes, &bias, &err)); */

  // set kernel arguments
  int arg = 0;
  int mode = 0; //0 upsize, 1 downsize
  OCL_CHECK(err, err = kernel_flip.setArg(arg++, buffer_a));
  OCL_CHECK(err, err = kernel_flip.setArg(arg++, buffer_b));
  OCL_CHECK(err, err = kernel_flip.setArg(arg++, HOUT));
  OCL_CHECK(err, err = kernel_flip.setArg(arg++, WOUT));
  OCL_CHECK(err, err = kernel_flip.setArg(arg++, H));
  OCL_CHECK(err, err = kernel_flip.setArg(arg++, W));
  OCL_CHECK(err, err = kernel_flip.setArg(arg++, mode));

  //-----------------------------
  // Copy input data to device global memory
  std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_a}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");

//  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_k}, 0 /*0 means from host*/, NULL, &write_events[0]));
 // set_callback(write_events[0], "ooo_queue");

  //-----------------------------
  printf("Enqueueing NDRange kernel.\n");
  // This event needs to wait for the write buffer operations to complete
  // before executing. We are sending the write_events into its wait list to
  // ensure that the order of operations is correct.
  // Launch the Kernel
  std::vector<cl::Event> waitList;
  waitList.push_back(write_events[0]);
  OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_flip, 0, 1, 1, &waitList, &kernel_events[0]));
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
  cpu_print_out();



  //-----------------------------
  std::cout << "" << std::endl;
  std::cout << "All done" << std::endl;
  std::cout << "quit now" << std::endl;

  // exit
  return 0;
}
