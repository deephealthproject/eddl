// --------------------------------------------------------------------------------------------------------------
// FPGA kernels for EDDL Library - European Distributed Deep Learning Library.
// Version: 0.6.
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
// test_conv2D_8x8_resize. Test for the conv2D_8x8_resize kernel
//
// Constants:
//
//  - CPI = 8
//  - CPO = 8
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
#include "xcl2.hpp"

#include <ap_fixed.h>
#include <sys/time.h>

using std::vector;

// data type
//#define data_type float
#define data_type ap_fixed<8,4,AP_TRN,AP_WRAP>

// CL
cl::Buffer buf;
cl::Context context;
cl::CommandQueue q;
cl::Program program;
std::string binaryFile;

#define WMAX 256
#define HMAX 256
#define IMAX 512
#define OMAX 512

#define CPI 4
#define CPO 4

#define KW 3
#define KH 3

#define RELU 0  // Flag for ReLu activation. Active at high level
#define MULT 0  // Flag for ReLu activation. Active at high level
#define SIG 0  // Flag for ReLu activation. Active at high level
#define MAXP 1 // Flag for ReLu activation. Active at high level

#define KWmpool 2
#define KHmpool 2
#define SWmpool 2
#define SHmpool 2

int WO;
int HO;

int W;
int H;
int WIN; //W origin before resize
int HIN; //H origin before resize
int I;
int O;
int I_kernel;    // kernel must be at minimum 8 input channels 
int O_kernel;    // and 8 output channels
int GI;
int GO;
int MODE; // Flag for resize activation. 1 upsize, 2 downsize
float MULTIPLICADOR;
// buffers
data_type *data_in; //[  I * W * H        ]  __attribute__ ((__aligned__(16)));
data_type *data_origin; //[  I * WIN * HIN        ]  __attribute__ ((__aligned__(16)));
data_type *kernel;  //[  O * I * KW * KH  ]  __attribute__ ((__aligned__(16)));
data_type *bias;    //[  O                ]  __attribute__ ((__aligned__(16)));
data_type *out;     //[  O * W * H        ]  __attribute__ ((__aligned__(16)));
data_type *out_cpu; //[  O * W * H        ]  __attribute__ ((__aligned__(16)));
data_type *maxpool; //[  O * W * H        ]  __attribute__ ((__aligned__(16)));

void allocate_buffers() {
  data_in = (data_type*)malloc(I * W * H * sizeof(data_type));
  data_origin = (data_type*)malloc(I * WIN * HIN * sizeof(data_type));
  kernel = (data_type*)malloc(I_kernel * O_kernel * KW * KH * sizeof(data_type));
  bias = (data_type*)malloc(O * sizeof(data_type));
  out = (data_type*)malloc(O * W * H * sizeof(data_type));
  out_cpu = (data_type*)malloc(O * W * H * sizeof(data_type));
  maxpool = (data_type*)malloc(O * W * H * sizeof(data_type));
}

void parse_arguments(int argc, char **argv) {
  if (argc != 10) {
    printf("syntax:\n%s <XCLBIN File> <WIN> <HIN> <W> <H> <I> <O>  <MODE> <MULTIPLICADOR>\n", argv[0]);
    exit(1);
  }

  binaryFile = argv[1];  
  WIN = atoi(argv[2]);
  HIN = atoi(argv[3]);
  W = atoi(argv[4]);
  H = atoi(argv[5]);
  I = atoi(argv[6]);
  O = atoi(argv[7]);
  MODE = atoi(argv[8]);
  MULTIPLICADOR = atof(argv[9]);
  
  if(MAXP){
	  WO = (int) ((W - KWmpool) / SWmpool + 1);
	  HO = (int) ((H - KHmpool) / SHmpool + 1);
	  printf("WO %d HO %d", WO, HO);
  }else{
	  WO=W;
	  HO=H;
  }
  if (I < CPI) {
    I_kernel = CPI;
  } else {
    I_kernel = I;
    if ((I % CPI) != 0) {printf("Error, I must me multiple of %d or lower than %d\n", CPI, CPI); exit(1);}
  }
  
  if (MODE > 2 || MODE < 0){
	  printf("Error, Mode must be 0 for no resize, 1 for upsize or 2 for downsize\n"); exit(1);
  }
  
  if((MODE == 0 && HIN != H) || (MODE == 0 && WIN != W)){
	  printf("Error, Mode 0 there is no resize, sizes musy be equal H = Hin and W = Win\n"); exit(1);
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

void deallocate_buffers() {
  free(data_in);
  free(data_origin);
  free(kernel);
  free(bias);
  free(out);
  free(out_cpu);
  free(maxpool);
}


void resize(){
	if(MODE == 1){
		//printf("upsize\n");
			data_type valor;
			
			 data_type repetidos[256];
			 int pos_origin =0;
			 int pos_in =0;
			
			int Wrepeat = W/WIN; //las columnas
			int Hrepeat = H/HIN; //las filas
			
			int countW = 0;
			int countH = 0;
			int countHaux = 0;
			
			int enableW = 1;
			int enableH = 0;

			int contReptH = 0;
			int contReptW = 0;
			
			int faltanH = H - (Hrepeat * HIN);
			int faltanW = W - (Wrepeat * WIN);
			//printf("Wrepeat %d, Hrepeat %d, faltanW %d faltanH %d\n", Wrepeat, Hrepeat, faltanW, faltanH);

			for(int c = 0; c < I; c++){
				for(int i = 0; i<H; i++){
					for (int j = 0; j <W; j++){
						if(enableH){
							//repetimos fila
							data_in[pos_in] = repetidos[j];
							//printf(" Valor = %f Pos = %d enableH (%d,%d) contReptH = %d countH = %d", data_in[pos_in], pos_in,i,j,contReptH, countH);
							pos_in++;
							//printf("\n"); 
							if(j == W - 1){
								enableH = 0;
							}	
						}else{				
							//leemos nueva fila
							if(enableW){
								valor = data_origin[pos_origin];
								pos_origin++;
								enableW = 0;
							}
							//comprobamos si hay repetidos
							if(contReptW < Wrepeat){
								data_in[pos_in] = valor;
								//printf(" Valor = %f Pos = %d enableW 1 (%d,%d) contReptW = %d countW = %d\n", data_in[pos_in], pos_in,i,j, contReptW, countW);
								pos_in++;
								repetidos[j]=valor;
								contReptW++;
								if(j == W - 1){
									enableW = 1;
									contReptW = 0;
									countW = 0;
								}else if(contReptW == Wrepeat && countW == faltanW){
									enableW = 1;
									contReptW = 0;
								}	
							}else if(countW < faltanW){
								//se acaba la fila
								data_in[pos_in] = valor;
								pos_in++;
								//printf("enableW 3(%d,%d) countW = %d",i,j, countW);
								//printf("\n"); 
								countW = 0;
							}
						}
					}
					contReptH++;
					if(!enableH && (contReptH < Hrepeat)){
						//printf("i = %d contReptH = %d countH = %d\n", i, contReptH, countH);
						enableH = 1;
					}else{
						//printf("FINAL H i = %d contReptH = %d countH = %d\n", i, contReptH, countH);
						contReptH = 0;
						enableH = 0;
						countHaux = countH;
					}
				}
				countH = 0;
				contReptH = 0;
				enableH = 0;
				countHaux = 0;
			}
			
	}else if (MODE == 2){
		//printf("downsize\n");
		int Wstride = 1 + ((WIN-1)/ W);
		int Hstride = 1 + ((HIN-1)/ H);
		int contHOUT = 0;
		int contWOUT = 0;
		int add = 0;
		int pos = 0;

		for(int c = 0; c < I; c++){
			for(int i = 0; i<HIN; i++){
				for (int j = 0; j <WIN; j++){
					
					data_type valor = data_origin[(i*WIN+j)+ add];
					
					if((i % Hstride == 0) || ((i == HIN-1) && (contHOUT < H))){
						if ((j % Wstride ==0) || ((j == WIN-1) && (contWOUT < W))){					
							data_in[pos] = valor;
							pos++;
							contWOUT++;
						}
						if(j==WIN-1){
							contHOUT++;
						}			
					}		
				}
				contWOUT=0;
			}
			add+=HIN*WIN;
			contHOUT=0;
		}
	
	}else{
		for(int i = 0; i < HIN*WIN*I; i++){
			data_in[i] = data_origin[i];
		}
		
	}


}

void mult(){
	int size = I * WIN * HIN;
	printf("MUltiplicador %f\n", MULTIPLICADOR);
	for (int i = 0; i < size; ++i) data_origin[i] *= (data_type)MULTIPLICADOR;
}

void sigmoid(){
	int size = O * WO * HO;
	data_type valor;
	for (int i = 0; i < size; i++) {
		valor = 1/(1+exp(-(double)out_cpu[i]));
		out_cpu[i] = valor;
	}  // check exp
}

void maxpool_test() {

	for (int i=0; i < O; i++) {
			   for (int ho=0; ho < HO; ho++) {
					for (int wo=0; wo < WO; wo++) {
						int idx_out = wo*WO+ho + (HO * WO * i);
						data_type max = -INFINITY;

						for (int kw=0; kw < KWmpool; kw++) {
							for (int kh=0; kh < KHmpool; kh++) {
								int idx_h = (ho * SHmpool)  + kh;
								int idx_w = (wo * SWmpool) + kw;
								int index = idx_h + (W * idx_w);
								int addr = index + (H * W * i);
							 
								data_type val = maxpool[addr];
								if (val > max) max = val;
								//printf("idx_out = %d addr = %d val = %f max = %f\n",idx_out, addr, val, max);
							 }
						}
							 
						out_cpu[idx_out] = max;
					}
				}
	}
}

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

void cpu_print_out_cpu() {
  printf("\nout cpu:\n");
  for (int c=0; c<O; c++) {
    printf(" channel %d:\n", c);
    printf("   ");
    for (int h=0; h<HO; h++) {
      for (int w=0; w<WO; w++) {
      	// data_in pixel position
	      int addr_p = (c * WO * HO) + (h * WO) + w;
        printf("%6.2f ", float(out_cpu[addr_p]));
      }
      printf("\n");
      printf("   ");
    }
    printf("\n");
  }
}

void cpu_print_maxpool() {
  printf("ANTES MAXPOOL:\n");
  for (int c=0; c<O; c++) {
    printf(" channel %d:\n", c);
    printf("   ");
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
      	// data_in pixel position
	      int addr_p = (c * W * H) + (h * W) + w;
        printf("%6.2f ", float(maxpool[addr_p]));
      }
      printf("\n");
      printf("   ");
    }
    printf("\n");
  }
}


void cpu_print_data_origin() {
  printf("data origin:\n");
  for (int c=0; c<I; c++) {
    printf(" channel %d:\n", c);
    printf("   ");
    for (int h=0; h<HIN; h++) {
      for (int w=0; w<WIN; w++) {
      	// data_in pixel position
	      int addr_p = (c * WIN * HIN) + (h * WIN) + w;
        printf("%6.2f ", data_origin[addr_p]);
      }
      printf("\n");
      printf("   ");
    }
    printf("\n");
  }
}

void cpu_conv2d() {

  int size_out = O * W * H;
  for (int i=0; i<size_out; i++) {
	maxpool[i] = 0.f;
	data_in[i] = 0.f;
  }
  
  //cpu_print_data_origin();
  mult();
  //cpu_print_data_origin();
  resize();
  //cpu_print_data_in();

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
	            if (!padding) maxpool[addr_o] += data_in[addr_p] * kernel[addr_k];
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
        maxpool[addr_o] += bias[cout];
      }
    }
  }

  //añadimos relu
  if(RELU){
    for (int cout=0; cout<O; cout++) {
      for (int h=0; h<H; h++) {
        for (int w=0; w<W; w++) {
          int addr_o = (h * W * O) + (w * O) + cout;
          if (maxpool[addr_o] < 0.0) maxpool[addr_o] = 0.0;
        }
      }
    }
  }
 cpu_print_maxpool();
  if(MAXP){
	  maxpool_test();
  cpu_print_out_cpu();
  }
  sigmoid();
  //cpu_print_out_cpu();
}



void cpu_print_kernels() {
  printf("kernels:\n");
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
            printf("%6.2f ", float(kernel[addr_k]));
      	  }
	        printf("\n");
        }
      }
    }
  }
}

void cpu_print_bias() {
  printf("bias:\n");
  for (int cout=0; cout<O; cout++) {
    printf("%6.2f ", float(bias[cout]));
  }
  printf("\n");
}

void cpu_print_out() {
  printf("output: cpu (fpga)\n");
  for (int cout=0; cout<O; cout++) {
    printf("channel %d:\n", cout);
    for (int h=0; h<HO; h++) {
      for (int w=0; w<WO; w++) {
	      // data_out pixel position
        int addr_o = (cout * WO * HO) + (h * WO) + w;
        printf(" %10.6f (%10.6f) (diff %10.6f) | ", float(out_cpu[addr_o]), float(out[addr_o]), float(out_cpu[addr_o]-out[addr_o]));
      }
      printf("\n");
    }
  }
}

void check_result() {

  int error = 0;
  for (int cout=0; cout<O; cout++) {
    for (int h=0; h<HO; h++) {
      for (int w=0; w<WO; w++) {
      	// data_out pixel position
        int addr_o = (cout * WO * HO) + (h * WO) + w;
        if (fabs(float(out_cpu[addr_o]) - float(out[addr_o])) > 0.001) { //fabs(float(out_cpu[addr_o]) - float(out[addr_o])) > 0.001 
          printf("Results mismatch at cout %d h %d w %d: %6.4f %6.4f (diff %6.4f)\n", cout, h, w, float(out_cpu[addr_o]), float(out[addr_o]), fabs(float(out_cpu[addr_o]-out[addr_o])));
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

  parse_arguments(argc, argv);

  printf("Test CONV: [IxWxH] = [%dx%dx%d] -> [OxWxH] = [%dx%dx%d] (kernel [%dx%d], stride [1x1], padding [1x1])\n", I, WIN, HIN, O, W, H, KW, KH);

  cl_int err;
  cl::Kernel kernel_conv2d;

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

  OCL_CHECK(err, kernel_conv2d = cl::Kernel(program,"k_conv2D_4x4_resize", &err));
  std::cout << "Kernel sucessfully created" << std::endl ;

  size_t size_data_origin_bytes = WIN * HIN * I * sizeof(data_type);
  size_t size_output_in_bytes = WO * HO * O * sizeof(data_type);
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
    for (int h=0; h<HIN; h++) {
      for (int w=0; w<WIN; w++) {
          data_type value = data_type(i); //data_type((i * W * H) + (h * W) + w); //c+1; // (data_type)((c * 25) + (h * W) + w);
          data_origin[addr] = dist(gen);
          addr++;
      }
    }
  }

 //cpu_print_data_origin();
 
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
          if ((i<I) && (o<O)) kernel[addr_k] = dist(gen); //value; //dist(gen);
          else kernel[addr_k] = 0;
        }
      }
      if ((o < O) && (i < I)) kernel_id++;
    }
  }

  std::cout << "Filling bias buffer with useful data" << std::endl;
  for (int cout=0; cout<O; cout++) bias[cout] = dist(gen);

  //-----------------------------
  // THIS PAIR OF EVENTS WILL BE USED TO TRACK WHEN A KERNEL IS FINISHED WITH
  // THE INPUT BUFFERS. ONCE THE KERNEL IS FINISHED PROCESSING THE DATA, A NEW
  // SET OF ELEMENTS WILL BE WRITTEN INTO THE BUFFER.
  vector<cl::Event> kernel_events(1);
  vector<cl::Event> read_events(1);
  vector<cl::Event> write_events(3);
  cl::Buffer buffer_a;
  cl::Buffer buffer_b;
  cl::Buffer buffer_k;
  cl::Buffer buffer_bias;

  //-----------------------------
  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  std::cout << "Creating Buffers..." << std::endl;

  OCL_CHECK(err, buffer_a = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_data_origin_bytes, data_origin, &err));
  OCL_CHECK(err, buffer_b = cl::Buffer(context, CL_MEM_WRITE_ONLY  | CL_MEM_USE_HOST_PTR , size_output_in_bytes, out, &err));
  OCL_CHECK(err, buffer_k = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_kernel_in_bytes, kernel, &err));
  OCL_CHECK(err, buffer_bias = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR , size_bias_in_bytes, bias, &err));

  //-----------------------------
  // Copy input data to device global memory
  // std::cout << "Copying data (Host to Device)..." << std::endl;
  // Because we are passing the write_events, it returns an event object
  // that identifies this particular command and can be used to query
  // or queue a wait for this particular command to complete.
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_a}, 0 /*0 means from host*/, NULL, &write_events[0]));
  set_callback(write_events[0], "ooo_queue");

  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_k}, 0 /*0 means from host*/, NULL, &write_events[1]));
  set_callback(write_events[1], "ooo_queue");

  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_bias}, 0 /*0 means from host*/, NULL, &write_events[2]));
  set_callback(write_events[2], "ooo_queue");  
  
  // timint stats
  unsigned long long prof_time;
  struct timeval prof_t1;
  gettimeofday(&prof_t1, NULL);

  int O_ITER = (O + (CPO-1)) / CPO;
  int I_ITER = (I + (CPI-1)) / CPI;
  int enable_relu = RELU;
  int enable_mult = MULT;
  int enable_maxpool = MAXP;
  int enable_sigmoid = SIG;

  // set kernel arguments
  int arg = 0;
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_a));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_a));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_a));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_a));
  
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, H));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, W));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, HIN));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, WIN));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, HO));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, WO));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, I));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, O));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, I_ITER));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, O_ITER));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, MULTIPLICADOR));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, enable_relu));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, enable_mult));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, enable_maxpool));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, enable_sigmoid));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, MODE));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_k));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_bias));
  
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_b));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_b));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_b));
  OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, buffer_b));

	cl_ulong time_start, time_end;
  //-----------------------------
  // printf("Enqueueing NDRange kernel.\n");
  // This event needs to wait for the write buffer operations to complete
  // before executing. We are sending the write_events into its wait list to
  // ensure that the order of operations is correct.
  // Launch the Kernel
  std::vector<cl::Event> waitList;
  waitList.push_back(write_events[0]);
  waitList.push_back(write_events[1]);
  waitList.push_back(write_events[2]);
  OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_conv2d, 0, 1, 1, &waitList, &kernel_events[0]));
  set_callback(kernel_events[0], "ooo_queue");

  OCL_CHECK(err, err = kernel_events[0].wait());

  // timing
/*   struct timeval prof_t2;
  gettimeofday(&prof_t2, NULL);
  prof_time = ((prof_t2.tv_sec - prof_t1.tv_sec) * 1000000) + (prof_t2.tv_usec - prof_t1.tv_usec);
  printf("Timing: %8lld usec\n", prof_time); */

	kernel_events[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
	kernel_events[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
	double diff = time_end-time_start;
	std::cout<< "TIME KERNEL = " << (diff/1000000)<<" ms \n"<<std::endl;


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

  std::cout << "computing conv in CPU..." << std::endl;

  //cpu_print_data_in();
  //cpu_print_kernels();
  //cpu_print_bias();
  cpu_conv2d();
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
