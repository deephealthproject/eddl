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

// --------------------------------------------------------------------------------------------------------------
//
// Convolution kernel
//
// Description
// This kernel computes the convolution operation for a given set of output
// channels. The kernel has a defined set of input channels (CPI) and output
// channels (CPO) where the convolution is performed in parallel.
// The kernel receives the input geometry (I, W, H) as arguments and performs
// the convolution over CPO channels. For I>CPI configurations the kernel iterates on the
// input channels to produce the output channels. For O>CPO the kernel iterates for every group of CPO
// output channels.
// The kernel uses DataFlow model and is optimized in order to be bounded by the memory bandwidth.
//
//  Dataflow:
//
//   .......                                                                                                ........
//   |     | ---> read_bias ----------------------------------------------                                  |      |
//   |     |                                                             |                                  |      |
//   |     | ---> read_kernel ------------------------------------       |                                  |      |
//   |     |                      .....                          |       |      .....                       |      |
//   |     | ---> read_channel -> |   |                          |       |      |   | --> write_channel --> |      |
//   |     | ---> read_channel -> |   |   ..........   .....   .....   .....    |   | --> write_channel --> |      |
//   |  D  | ---> read_channel -> | j |   |        |   |   |   |   |   |   |    | s | --> write_channel --> |   D  |
//   |  D  | ---> read_channel -> | o | ->| padding|-> |cvt| ->|mul| ->|add| -> | p | --> write_channel --> |   D  |
//   |  R  | ---> read_channel -> | i |   |        |   |   |   |   |   |   |    | l | --> write_channel --> |   R  |
//   |     | ---> read_channel -> | n |   ..........   .....   .....   .....    | i | --> write_channel --> |      |
//   |     | ---> read_channel -> |   |                                         | t | --> write_channel --> |      |
//   |     | ---> read_channel -> |   |                                         |   | --> write_channel --> |      |
//   |     |                      .....                                         .....                       |      |
//   |     |                                                                                                |      |
//   .......                                                                                                ........
//
// The kernels asumes the following memory allocation for data:
//    - input data : I x H x W
//    - kernels    : GO x GI x CPO x CPI x KH x KW
//    - bias       : O
//    - output data: O x H x W
//
// (GI = group of input channels, GO = group of output channels)
// (I = GI x CPI, O = GO x CPO)
//
// Fixed (static) parameters:
//    - CPI: Number of input channels supported in one iteration of the kernel
//    - CPO: Number of output channels supported in one iteration of the kernel
//    - KH, KW: Kernel size (3x3)
//    - PH, PW: Padding (1x1) (implicit in the code)
//    - SH, SW: Stride (1x1) (implicit in the code)
//    - WMAX: Maximum value of the width of an input channel
//    - WHMAX: Maximum value of the width multiplied by the height of an input channel
//
// Arguments:
//    - I: Number of input channels
//    - O: Number of output channels
//    - I_ITER: Number of input iterations, which means ceil(I / CPI)
//    - O_ITER: Number of output iterations, which means ceil(O / CPO)
//    - W: Channel width
//    - H: Channel height
//    - ptr_data (x8): Memory pointers to input data
//    - ptr_kernel: Memory pointer to kernels
//    - ptr_bias: Memory pointer to bias
//    - ptr_out (x8): Memory pointers to output buffer
//
//

// Headers
#include <math.h>
#include <stdio.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// Enable this define to get information (sw_emu)
//#define DEBUG_VERBOSE

extern "C" {

// Data type to be used
#define data_type float

// To allow using defines inside Xilinx pragmas
#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

// Fixed parameters (optimized at compilation/synthesis time)
#define KW       3  // kernel width
#define KH       3  // kernel height
#define CPI      4  // channels per input port
#define CPO      4  // channels per output port

// Maximum width and width*height
#define WMAX 256
#define WHMAX 256*256

// Data type for input data to the conv module
struct pixel_in_t {           // pixel in
  data_type pixel[CPI];
};

// Data type for output data from the conv module
struct pixel_out_t {          // pixel out
  data_type pixel[CPO];
};

// frames struct (KWxKH)
struct frame_t {
  pixel_in_t pixel[9];
};

// blocks to read/write from/to memory
#define BLOCK_SIZE 4              // block size is the number of pixels to read/write per cycle
struct block_t {
  data_type pixel[BLOCK_SIZE];
};

// ---------------------------------------------------------------------------------------
// read_bias. Reading bias from memory and sending to add module.
//
// Arguments:
//   b_ptr               : pointer to bias
//   offset_bias         : offset to bias
//   b_out               : output stream
//
// All the bias are read and sent through the out stream
//
static void read_bias(int offset_bias, data_type *b_ptr, hls::stream<pixel_out_t> &b_out) {

  #ifdef DEBUG_VERBOSE
  printf("read_bias: start\n");
  #endif

  pixel_out_t bias;
  #pragma HLS ARRAY_PARTITION variable=bias dim=0

  // we read the bias
  for (int i=0; i<CPO; i++) {
    data_type v = b_ptr[i + offset_bias];
    bias.pixel[i] = v;
  }

  #ifdef DEBUG_VERBOSE
  printf("bias read: ");
  for (int c=0; c<CPO; c++) printf(" %f ", float(bias.pixel[c]));
  printf("\n");
  #endif

  b_out << bias;

  #ifdef DEBUG_VERBOSE
  printf("read_bias: end\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// read_kernel. Reads kernels and sends them through the stream
//
// Arguments:
//   I_ITER              : Number of input iterations (I / CPI)
//   k_ptr               : pointer to kernels
//   offset_kernel       : offset to kernels
//   k_out               : output stream
//
// kernels are stored in memory with the format GO x GI x CPO x CPI x KH x KW
// This storage formats lets the module to read memory sequentially and send all the
// kernels in the same order they are read through the output stream.
// kernels are sent in frame structures (3x3 grid)
//
static void read_kernel(int I_ITER, int offset_kernel, data_type *k_ptr, hls::stream<frame_t> &k_out){

  #ifdef DEBUG_VERBOSE
  printf("read_kernel: start\n");
  #endif

  // we read all the kernels and send them through the stream
  frame_t frame_k;
  #pragma HLS ARRAY_PARTITION variable=frame_k dim=0
  int cpi = 0;
  int p = 0;

  int size = KW * KH * CPO * I_ITER * CPI;
  read_kernel_loop:
  for (int i=0; i<size; i++) {
    frame_k.pixel[p].pixel[cpi] = k_ptr[i+ offset_kernel];
    p = p + 1;
    if (p == 9) {
      p = 0;
      cpi = cpi+1;
      if (cpi == CPI) {
        cpi = 0;
	      k_out << frame_k;
        #ifdef DEBUG_VERBOSE
	      printf("kernel read:\n");
	      for (int c=0; c<CPI; c++) {
          printf("channel %d: ", c);
	        for (int p=0; p<9; p++) printf(" %f ", float(frame_k.pixel[p].pixel[c]));
	        printf("\n");
	      }
        #endif
      }
    }
  }

  #ifdef DEBUG_VERBOSE
  printf("read_kernel: end\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// read_data_channel. Reads one data channel and sends it through the stream
//
// Arguments:
//   H, W                : Data channel height and width
//   I_ITER              : Number of input iterations (I / CPI)
//   ptr                 : pointer to input data
//   offset              : offset within input data
//   out                 : output stream
//   enable              : if not set the module produces just zeros and does not read memory
//   id                  : module identifier (for debug purposes in sw_emu mode)
//
// If I_ITER > 1 the module reads several input channels. An stride between read channels
// is computed. As there are eight read_data_channel modules all of them read input channels
// in an interleaved manner (module 0 reads channels 0, 8, 16, 32..., module 1 reads channels
// 1, 9, 17, 33, ..., and so on)
//
static void read_data_channel(int H, int W, int I_ITER, block_t *ptr, int offset, hls::stream<block_t> &out, int enable, int id) {

  int offset_read = offset;

  #ifdef DEBUG_VERBOSE
  printf("read_data_channel_%d starts\n", id);
  #endif

  for (int i_iter = 0; i_iter < I_ITER; i_iter++) {
    if (enable) {
      read_data_channel_loop:
      for (int r=0; r<H*W/BLOCK_SIZE; r++) {
        #pragma HLS PIPELINE II=1
        block_t data;
        data = ptr[r+offset_read];
        #ifdef DEBUG_VERBOSE
        printf("data read:");
        for (int x=0; x<BLOCK_SIZE; x++) printf(" %f", float(data.pixel[x]));
        printf("\n");
        #endif
        out << data;
      }
      // next channel (0, 8, 16, 32, ...)
      offset_read = offset_read + ((H*W)*CPI)/BLOCK_SIZE;
    } else {
      read_data_channel_zeros_loop:
      block_t data_zeros;
      for (int b=0; b<BLOCK_SIZE; b++) {
        #pragma HLS UNROLL
        data_zeros.pixel[b] = 0;
      }
      for (int r=0; r<H*W/BLOCK_SIZE; r++) {
        #pragma HLS PIPELINE II=1
        out << data_zeros;
      }
    }
  }
  #ifdef DEBUG_VERBOSE
  printf("read_data_channel_%d ends\n", id);
  #endif
}

// ---------------------------------------------------------------------------------------
// join. Joins input streams of pixels and combines them to produce groups of pixels
//
// Arguments:
//   H, W                : Data channel height and width
//   I_ITER              : Number of input iterations (I / CPI)
//   in0, ... in7        : input streams
//   out                 : output stream
//
// The input streams have width of BLOCK_SIZE elements whereas the output stream
// has width of CPI elements. This module gets the first elements of all input
// streams and produces an output data, then it takes the second elements of all
// input streams and produces a new output data, and so on... For every received
// input data from all streams the join module uses BLOCK_SIZE cycles to produce
// BLOCK_SIZE data items. All data items are sent through the output stream
//
static void join(int H, int W, int I_ITER,
                               hls::stream<block_t> &in0, hls::stream<block_t> &in1, hls::stream<block_t> &in2, hls::stream<block_t> &in3,
                               hls::stream<pixel_in_t> &out) {

  #ifdef DEBUG_VERBOSE
  printf("join starts\n");
  #endif

  for (int i_iter = 0; i_iter < I_ITER; i_iter++) {

    join_loop:
    for (int r=0; r<H*W/BLOCK_SIZE; r++) {
      #pragma HLS PIPELINE II=1
      pixel_in_t data;
      block_t b0 = in0.read();
      block_t b1 = in1.read();
      block_t b2 = in2.read();
      block_t b3 = in3.read();
      #ifdef DEBUG_VERBOSE
      printf("read for r %d completed\n", r);
      #endif
      for (int b=0; b<BLOCK_SIZE; b++) {
        data.pixel[0] = b0.pixel[b];
        data.pixel[1] = b1.pixel[b];
        data.pixel[2] = b2.pixel[b];
        data.pixel[3] = b3.pixel[b];
        out << data;
      }
    }

  }
  #ifdef DEBUG_VERBOSE
  printf("join starts\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// split. Splits incomming pixels grouped in pixel_out_t struct into eight output streams
// of size BLOCK_SIZE elements each.
//
// Arguments:
//   H, W                : sata channel height and width
//   in                  : input stream
//   out0, ... out7      : output streams
//
// The input stream has CPO pixels per data item whereas each output stream has
// BLOCK_SIZE pixels per data item. Therefore, this module reads during BLOCK_SIZE
// cycles the input stream and assigns each pixel from each read data item into
// every output data item to be sent. After those cycles the out data items are
// sent through the corresponding output stream
//
static void split(int H, int W, hls::stream<pixel_out_t> &in, hls::stream<block_t> &out0, hls::stream<block_t> &out1, hls::stream<block_t> &out2, hls::stream<block_t> &out3) {
  #ifdef DEBUG_VERBOSE
  printf("split starts\n");
  #endif

  split_loop:
  for (int r=0; r<H*W/BLOCK_SIZE; r++) {
    #pragma HLS PIPELINE II=1
    pixel_out_t data;
    block_t b0, b1, b2, b3/*, b4, b5, b6, b7*/;
    for (int b=0; b<BLOCK_SIZE; b++) {
      data = in.read();
      b0.pixel[b] = data.pixel[0];
      b1.pixel[b] = data.pixel[1];
      b2.pixel[b] = data.pixel[2];
      b3.pixel[b] = data.pixel[3];
    }
    out0 << b0;
    out1 << b1;
    out2 << b2;
    out3 << b3;
  }

  #ifdef DEBUG_VERBOSE
  printf("split ends\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// write_data_channel. Writes one data channel from the elements read from an input stream
//
// Arguments:
//   H, W                : Data channel height and width
//   ptr                 : pointer to output buffer
//   offset              : offset within input buffer
//   in                  : input stream
//   enable              : if not set the module just consumes the input stream and does not write memory
//   id                  : module identifier (for debug purposes in sw_emu mode)
//
// On every cycle the module receives BLOCK_SIZE pixels to write into memory
//
static void write_data_channel(int H, int W, block_t *ptr, int offset, hls::stream<block_t> &in, int enable, int id) {
  #ifdef DEBUG_VERBOSE
  printf("write_data_channel_%d starts\n", id);
  #endif

  if (enable) {
    write_data_channel_loop:
    for (int r=0; r<H*W/BLOCK_SIZE; r++) {
      #pragma HLS PIPELINE II=1
      block_t data;
      data = in.read();
      ptr[r+offset] = data;
    }
  } else {
    for (int r=0; r<H*W/BLOCK_SIZE; r++) {
      #pragma HLS PIPELINE II=1
      block_t data;
      data = in.read();
    }
  }
  #ifdef DEBUG_VERBOSE
  printf("write_data_channel_%d ends\n", id);
  #endif
}
// -------------------------------------------------------------------------------
// relu: module of ReLu function
//
// Arguments:
//   enable_relu: : Flag to enable ReLu function
//   H            : Height of the input channel
//   W            : Width of the input channel
//   in           : input data stream
//   out          : output data stream
//
// This module builds ReLu function by instantiatig streams and
// building the dataflow model with the corresponding modules
//
static void relu(int enable_relu, int H, int W, hls::stream<pixel_out_t> &in, hls::stream<pixel_out_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("relu: start\n");
#endif

  pixel_out_t data;
  int data_size = W * H;
  for (int i=0; i < data_size; i++) {
    #pragma HLS PIPELINE II=1
    data  = in.read();
    for(int cpo = 0; cpo<CPO; cpo++){
      #pragma HLS UNROLL
      if(enable_relu & (data.pixel[cpo] < 0)) data.pixel[cpo] = data_type(0.f);
    }
      out << data;
    }

#ifdef DEBUG_VERBOSE
  printf("relu: end\n");
#endif
}
// ---------------------------------------------------------------------------------------
// padding. Adds padding to the input and forwards it through the output
//
// Arguments:
//   H                 : Height of input channel
//   W                 : Width of input channel
//   I_ITER            : Number of input iterations (I / CPI)
//   in                : input stream
//   out               : output stream
//
static void padding(int H, int W, int I_ITER, hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out) {

  #ifdef DEBUG_VERBOSE
  printf("padding: start\n");
  #endif

  pixel_in_t data;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=data complete)

  pixel_in_t zero;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=zero complete)

  padding_cpi_loop:
  for (int cpi=0; cpi<CPI; cpi++) zero.pixel[cpi] = 0.f;

  padding_iter_loop:
  for(int iter = 0; iter < I_ITER; iter++){
    padding_h_loop:
    for(int h = 0; h < H + 2; h++){
      #pragma HLS_PIPELINE II=1
      padding_w_loop:
      for(int w = 0; w < W + 2; w++){
        #pragma HLS_PIPELINE II=1
        if (h==0 || h == H+1 || w == 0 || w == W+1) {
          data = zero;
        } else {
          data = in.read();
        }
        #ifdef DEBUG_VERBOSE
        for(int cpi = 0;cpi<CPI;cpi++) printf("data.pixel[%d] = %6.2f  ", cpi, float(data.pixel[cpi]));
        printf("\n");
        #endif
        out << data;
      }
    }
  } // iter

  #ifdef DEBUG_VERBOSE
  printf("padding: end\n");
  #endif
}

// ---------------------------------------------------------------------------------------------------
// cvt: reads an input stream with an image of format (H, W, CPI) and writes an output stream
// in a 2D format based on (KW, KH). (SW=1, SH=1) stride is assumed and (PW=1, PH=1) padding is assumed.
// The function outputs data in the format (KH, KW, CPI).
//
// Arguments:
//   H      : Height of input channel
//   W      : Width of input channel
//   I_ITER : Number of input iterations (I / CPI)
//   in     : input stream (format pixel_in_t)
//   out    : output stream (format frame_t)
//
static void cvt(int H, int W, int I_ITER, hls::stream<pixel_in_t> &in, hls::stream<frame_t> &out) {

  #ifdef DEBUG_VERBOSE
  printf("cvt: start\n");
  #endif

  cvt_i_iter_loop:
  for(int i_iter = 0; i_iter < I_ITER; i_iter++){

    // Now we process the input data and convert the data into frames
    // buffers (keep three rows)
    pixel_in_t buffer0[WMAX+2];
    pixel_in_t buffer1[WMAX+2];
    pixel_in_t buffer2[WMAX+2];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=buffer0 cyclic dim=1 factor=CPI)
    DO_PRAGMA(HLS ARRAY_PARTITION variable=buffer1 cyclic dim=1 factor=CPI)
    DO_PRAGMA(HLS ARRAY_PARTITION variable=buffer2 cyclic dim=1 factor=CPI)

    // frame
    frame_t frame;
    DO_PRAGMA(HLS ARRAY_PARTITION variable=frame)

    // We loop for every incoming pixel
    cvt_loop_1:
    for (int pin_row=0; pin_row < H+2; pin_row++) {
      cvt_loop_2:
      for (int pin_col=0; pin_col < W+2; pin_col++) {
        // get the pixel
        pixel_in_t pixel;
        pixel = in.read();
        // row buffer write (in which buffer row we write the pixel)
        int row0_buffer_write = (pin_row % 3) == 0;
        int row1_buffer_write = (pin_row % 3) == 1;
        // first row buffer
        int row0 = (pin_row <= 2) | ((pin_row % 3) == 2);
        int row1 = !row0 & ((pin_row % 3) == 0);
        // we write the pixel into the buffer
        if (row0_buffer_write) buffer0[pin_col] = pixel; else if (row1_buffer_write) buffer1[pin_col] = pixel; else buffer2[pin_col] = pixel;
        // build the frame
        pixel_in_t p0, p1, p2, p3, p4, p5, p6, p7, p8;
        int shift_frame = (pin_row>1) & (pin_col > 2);
        int send_frame = (pin_row>1) & (pin_col > 1);
        pixel_in_t pixel_b0, pixel_b1, pixel_b2;
        pixel_b0 = buffer0[pin_col];
        pixel_b1 = buffer1[pin_col];
        pixel_b2 = buffer2[pin_col];
        // p0, p1, p2
        if (shift_frame) {p0 = p1;} else if (pin_col==0) {if (row0) p0 = pixel_b0; else if (row1) p0 = pixel_b1; else p0 = pixel_b2;}
        if (shift_frame) {p1 = p2;} else if (pin_col==1) {if (row0) p1 = pixel_b0; else if (row1) p1 = pixel_b1; else p1 = pixel_b2;}
        if (row0) p2 = pixel_b0; else if (row1) p2 = pixel_b1; else p2 = pixel_b2;
        // p3, p4, p5
        if (shift_frame) {p3 = p4;} else if (pin_col==0) {if (row0) p3 = pixel_b1; else if (row1) p3 = pixel_b2; else p3 = pixel_b0;}
        if (shift_frame) {p4 = p5;} else if (pin_col==1) {if (row0) p4 = pixel_b1; else if (row1) p4 = pixel_b2; else p4 = pixel_b0;}
        if (row0) p5 = pixel_b1; else if (row1) p5 = pixel_b2; else p5 = pixel_b0;
        // p6, p7, p8
        if (shift_frame) {p6 = p7;} else if (pin_col==0) {if (row0) p6 = pixel_b2; else if (row1) p6 = pixel_b0; else p6 = pixel_b1;}
        if (shift_frame) {p7 = p8;} else if (pin_col==1) {if (row0) p7 = pixel_b2; else if (row1) p7 = pixel_b0; else p7 = pixel_b1;}
        if (row0) p8 = pixel_b2; else if (row1) p8 = pixel_b0; else p8 = pixel_b1;

        if (send_frame) {
          frame.pixel[0] = p0; frame.pixel[1] = p1; frame.pixel[2] = p2;
          frame.pixel[3] = p3; frame.pixel[4] = p4; frame.pixel[5] = p5;
          frame.pixel[6] = p6; frame.pixel[7] = p7; frame.pixel[8] = p8;
          out << frame;
          #ifdef DEBUG_VERBOSE
          printf("cvt: frame sent:\n");
          for (int cpi=0; cpi<CPI; cpi++) {
            printf("  cpi %d:\n", cpi);
            printf("    %6.4f %6.4f %6.4f\n", float(frame.pixel[0].pixel[cpi]), float(frame.pixel[1].pixel[cpi]), float(frame.pixel[2].pixel[cpi]));
            printf("    %6.4f %6.4f %6.4f\n", float(frame.pixel[3].pixel[cpi]), float(frame.pixel[4].pixel[cpi]), float(frame.pixel[5].pixel[cpi]));
            printf("    %6.4f %6.4f %6.4f\n", float(frame.pixel[6].pixel[cpi]), float(frame.pixel[7].pixel[cpi]), float(frame.pixel[8].pixel[cpi]));
          }
          #endif
        }
      }
    }
  } //i_iter

  #ifdef DEBUG_VERBOSE
  printf("cvt: end\n");
  #endif
}

// ----------------------------------------------------------------------------------------
// mul: This function performs the multiplication of an input frame with the stored kernels
// and sends the produced pixels. Before normal operation it receives its kernels
// Arguments:
//   H     : Height of the input channel
//   W     : Width of the input channel
//   I_ITER: Number of input iterations (I / CPI)
//   in    : input stream with incoming data frames
//   k_in  : input stream with kernels
//   out   : output stream
//
static void mul(int H, int W, int I_ITER, hls::stream<frame_t> &in, hls::stream<frame_t> &k_in, hls::stream<pixel_out_t> &out) {

  #ifdef DEBUG_VERBOSE
  printf("mul: start\n");
  #endif

  frame_t kernel[CPO];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=kernel dim=0)
  frame_t data_in;

  // Reading the kernels
  mul_i_iter_loop:
  for(int i_iter = 0; i_iter < I_ITER; i_iter++){
    loop_mul_kernels_load_cpo:
    for (int cpo=0; cpo<CPO; cpo++) {
      #pragma HLS PIPELINE II=1
      kernel[cpo] = k_in.read();
    }
    #ifdef DEBUG_VERBOSE
    printf("mul: kernels received\n");
    for (int cpo=0; cpo < CPO; cpo++) {
      for (int cpi=0; cpi < CPI; cpi++) {
        printf("  cpi=%d, cpo=%d:\n", cpi, cpo);
        printf("    %6.4f %6.4f %6.4f\n", float(kernel[cpo].pixel[0].pixel[cpi]), float(kernel[cpo].pixel[1].pixel[cpi]), float(kernel[cpo].pixel[2].pixel[cpi]));
        printf("    %6.4f %6.4f %6.4f\n", float(kernel[cpo].pixel[3].pixel[cpi]), float(kernel[cpo].pixel[4].pixel[cpi]), float(kernel[cpo].pixel[5].pixel[cpi]));
        printf("    %6.4f %6.4f %6.4f\n", float(kernel[cpo].pixel[6].pixel[cpi]), float(kernel[cpo].pixel[7].pixel[cpi]), float(kernel[cpo].pixel[8].pixel[cpi]));
      }
    }
    #endif

    // now we read frames and produce the pixels
    data_type sum[CPO];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=sum dim=0 block factor=CPO)
    //
    int num_iterations = W * H;
    mul_sum_loop:
    for (int cpo=0; cpo<CPO; cpo++) sum[cpo] = 0.f;

    mul_num_iterations_loop:
    for (int i=0; i<num_iterations; i++) {
      data_in = in.read();

      #ifdef DEBUG_VERBOSE
      printf("mul: data received\n");
      for (int cpi=0; cpi<CPI; cpi++) {
        printf("  cpi=%d\n", cpi);
        printf("    %6.4f %6.4f %6.4f\n", float(data_in.pixel[0].pixel[cpi]), float(data_in.pixel[1].pixel[cpi]), float(data_in.pixel[2].pixel[cpi]));
        printf("    %6.4f %6.4f %6.4f\n", float(data_in.pixel[3].pixel[cpi]), float(data_in.pixel[4].pixel[cpi]), float(data_in.pixel[5].pixel[cpi]));
        printf("    %6.4f %6.4f %6.4f\n", float(data_in.pixel[6].pixel[cpi]), float(data_in.pixel[7].pixel[cpi]), float(data_in.pixel[8].pixel[cpi]));
      }
      #endif

      loop_mul_cpi:
      for (int cpi=0; cpi<CPI; cpi++) {
        #pragma HLS UNROLL
        loop_mul_j:
        for (int j=0; j<KW*KH; j++) {
          #pragma HLS UNROLL
          loop_mul_cpo:
          for (int cpo=0; cpo<CPO; cpo++) {
            #pragma HLS UNROLL
            sum[cpo] += data_in.pixel[j].pixel[cpi] * kernel[cpo].pixel[j].pixel[cpi];
          }
        }
      }

      pixel_out_t p_out;
      for (int cpo=0; cpo<CPO; cpo++) {
        #pragma HLS unroll
        #ifdef DEBUG_VERBOSE
        printf("mul: pixel produced cpo=%d -> %6.4f\n", cpo, float(sum[cpo]));
        #endif
        p_out.pixel[cpo] = sum[cpo];
        sum[cpo] = 0.f;
      }
      out << p_out;
    }
  } //i_iter

  #ifdef DEBUG_VERBOSE
  printf("mul: end\n");
  #endif
}

// -------------------------------------------------------------------------------
// add: This function performs the addition of all subpixels for the same channel.
// It adds also the corresponding bias.
//
// Arguments:
//   H     : Height of input channel
//   W     : Width of input channel
//   I_ITER: Number of input iterations (I / CPI)
//   in    : input streams data
//   b_in  : input stream bias
//   out   : output stream
//
static void add(int H, int W, int I_ITER, hls::stream<pixel_out_t> &in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

  #ifdef DEBUG_VERBOSE
  printf("add: start\n");
  #endif

  data_type bias[CPO];

  // number of iterations by CPI || CPO channels
  int num_iterations = W * H;

  // Buffer for all data and CPO channels
  data_type buff_o_channels[CPO][WHMAX];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=buff_o_channels dim=0 block factor=CPO)

  // We receive bias in packs of CPO
  pixel_out_t p_out;
  p_out = b_in.read();
  add_load_bias_loop:
  for (int b=0; b<CPO; b++) {
    #pragma HLS PIPELINE II=1
    bias[b] = p_out.pixel[b];
  }

  #ifdef DEBUG_VERBOSE
  for (int b=0; b<CPO; b++) {
    printf("Bias[%d] = %6.4f \n", b, float(bias[b]));
  }
  printf("add: bias received\n");
  for(int cpo = 0; cpo<CPO; cpo++){
    printf("Channel cpo = %d: ", cpo);
    for(int it = 0; it<num_iterations; it++){
      printf("%6.2f ", float(buff_o_channels[cpo][it]));
    }
    printf("\n");
  }
  #endif

  // All input data have effect into output add
  add_i_iter_loop:
  for (int i_iter = 0; i_iter < I_ITER; i_iter++){
    pixel_out_t data_out;
    #pragma HLS loop_flatten off
    add_load_data_it_loop:
    for(int it = 0; it<num_iterations; it++){
      pixel_out_t data_in;
      data_in = in.read();
      pixel_out_t data;
      add_load_data_cpo_loop:
      for (int cpo=0; cpo<CPO; cpo++) {
        #pragma HLS unroll
        if(i_iter == 0){
          data.pixel[cpo] = bias[cpo];
        } else {
          data.pixel[cpo] = buff_o_channels[cpo][it];
        }
        buff_o_channels[cpo][it] = data.pixel[cpo] + data_in.pixel[cpo];

        if(i_iter ==(I_ITER-1)){
          data_out.pixel[cpo] = buff_o_channels[cpo][it];
        }
      }
      if(i_iter ==(I_ITER-1)){
        out << data_out;
      }
    }
  } //i_iter

  #ifdef DEBUG_VERBOSE
  for (int cpo=0; cpo<CPO; cpo++) {
    printf("CH %d: ", cpo);
    for (int it=0; it<num_iterations; it++) {
      printf("%6.2f ", float(buff_o_channels[cpo][it]));
    }
    printf("\n");
  }
  #endif

  #ifdef DEBUG_VERBOSE
  printf("add: end\n");
  #endif
}

// -------------------------------------------------------------------------------
// conv: Convolutional kernel
//
// Arguments:
//   H      : Height of the input channel
//   W      : Width of the input channel
//   I_ITER : Number of input iterations (I / CPI)
//   in     : input data stream
//   k_in   : input kernel stream
//   b_in   : input bias stream
//   out    : output data stream
//
// This module builds the convolutional operation by instantiating streams and
// building the dataflow model with the corresponding modules
//
static void conv(int H, int W, int I_ITER, hls::stream<pixel_in_t> &in, hls::stream<frame_t> &k_in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

  // streams
  static hls::stream<pixel_in_t>  str_pad_cvt;  // padding->cvt
  static hls::stream<frame_t>     str_cvt_mul;  // cvt->mul
  static hls::stream<pixel_out_t> str_mul_add;  // mul->add

  // topology
  #pragma HLS dataflow
  padding(H, W, I_ITER, in, str_pad_cvt);            // padding
  cvt(H, W, I_ITER, str_pad_cvt, str_cvt_mul);       // cvt
  mul(H, W, I_ITER, str_cvt_mul, k_in, str_mul_add); // mul
  add(H, W, I_ITER, str_mul_add, b_in, out);         // add
}

// -------------------------------------------------------------------------------
// k_conv2D_K3x3_S1x1_P1x1_BS1
// Main kernel
//
// Arguments:
//   ptr_data (x8)      : pointers to input data
//   H                  : height of input channel
//   W                  : width of input channel
//   I                  : number of input channels
//   O                  : number of output channels
//   I_ITER             : input iterations, which means ceil(I / CPI)
//   O_ITER             : output iterations, which means ceil(O / CPO)
//   ptr_kernel         : ponter to kernels
//   ptr_bias           : pointer to bias
//   ptr_out (x8)       : pointers to output buffer
//
// This module creates the streams and builds the dataflow model using the specific modules
// defined above
//
void k_multiple_conv2D_4x4
                                (
		                             block_t *ptr_data0, block_t *ptr_data1, block_t *ptr_data2, block_t *ptr_data3,
                                 int H, int W, int I, int O, int I_ITER, int O_ITER, int enable_relu,
                                 data_type *ptr_kernel, data_type *ptr_bias,
                                 block_t *ptr_out0, block_t *ptr_out1, block_t *ptr_out2, block_t *ptr_out3
				) {

  #pragma HLS INTERFACE s_axilite port=W bundle=control
  #pragma HLS INTERFACE s_axilite port=H bundle=control
  #pragma HLS INTERFACE s_axilite port=I bundle=control
  #pragma HLS INTERFACE s_axilite port=O bundle=control
  #pragma HLS INTERFACE s_axilite port=I_ITER bundle=control
  #pragma HLS INTERFACE s_axilite port=O_ITER bundle=control
  #pragma HLS INTERFACE s_axilite port=enable_relu bundle=control
  #pragma HLS INTERFACE m_axi port=ptr_data0 offset=slave bundle=gmem0   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_data1 offset=slave bundle=gmem1  max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_data2 offset=slave bundle=gmem2  max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_data3 offset=slave bundle=gmem3  max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_kernel offset=slave bundle=gmem4 max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_bias offset=slave bundle=gmem5   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_out0  offset=slave bundle=gmem0   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_out1  offset=slave bundle=gmem1  max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_out2  offset=slave bundle=gmem2   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_out3  offset=slave bundle=gmem3  max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  // ptr_data struct to be packed as a single element vector (to improve memory read)
  // the compiler will do full structure access (all elements of structure)
  #pragma HLS data_pack variable = ptr_data0
  #pragma HLS data_pack variable = ptr_data1
  #pragma HLS data_pack variable = ptr_data2
  #pragma HLS data_pack variable = ptr_data3
  #pragma HLS data_pack variable = ptr_out0
  #pragma HLS data_pack variable = ptr_out1
  #pragma HLS data_pack variable = ptr_out2
  #pragma HLS data_pack variable = ptr_out3



  #ifdef DEBUG_VERBOSE
  printf("kernel starts...\n");
  #endif

  for (int o_iter = 0; o_iter<O_ITER; o_iter++) {

    #pragma HLS dataflow

    // we compute the enable_write signals
    int o_channel = (o_iter << 2);  // current output channel (first one in this iteration)
    int enable_write0 = o_channel < O;
    int enable_write1 = (o_channel + 1) < O;
    int enable_write2 = (o_channel + 2) < O;
    int enable_write3 = (o_channel + 3) < O;

    // we compute the enable_read signals
    int enable_read0 = (I >= 1);
    int enable_read1 = (I >= 2);
    int enable_read2 = (I >= 3);
    int enable_read3 = (I >= 4);


    // input and output streams
    static hls::stream<pixel_in_t> out_read_data;
    static hls::stream<frame_t> out_read_kernel;
    static hls::stream<pixel_out_t> out_read_bias;
    static hls::stream<pixel_out_t> out_conv;
    //ReLu stream
    static hls::stream<pixel_out_t> out_relu;

    static hls::stream<block_t> out_read_channel_0;
    static hls::stream<block_t> out_read_channel_1;
    static hls::stream<block_t> out_read_channel_2;
    static hls::stream<block_t> out_read_channel_3;


    static hls::stream<block_t> out_write_channel_0;
    static hls::stream<block_t> out_write_channel_1;
    static hls::stream<block_t> out_write_channel_2;
    static hls::stream<block_t> out_write_channel_3;

    // channel offsets for reading
    int channel_offset = (W * H) / BLOCK_SIZE;
    int offset_read_data_channel_0 = 0;
    int offset_read_data_channel_1 = channel_offset;
    int offset_read_data_channel_2 = channel_offset * 2;
    int offset_read_data_channel_3 = channel_offset * 3;


    // channel offsets for writing
    int offset_write_data_channel_0 = ((o_iter << 2) * channel_offset);
    int offset_write_data_channel_1 = channel_offset + ((o_iter << 2) * channel_offset);
    int offset_write_data_channel_2 = (channel_offset * 2) + ((o_iter << 2) * channel_offset);
    int offset_write_data_channel_3 = (channel_offset * 3) + ((o_iter << 2) * channel_offset);

    int offset_bias = o_iter << 2;
    int offset_kernel = o_iter * (I < 8 ? 8 : I) * 9 * CPO;

    read_data_channel(H, W, I_ITER, ptr_data0, offset_read_data_channel_0, out_read_channel_0, enable_read0, 0);
    read_data_channel(H, W, I_ITER, ptr_data1, offset_read_data_channel_1, out_read_channel_1, enable_read1, 1);
    read_data_channel(H, W, I_ITER, ptr_data2, offset_read_data_channel_2, out_read_channel_2, enable_read2, 2);
    read_data_channel(H, W, I_ITER, ptr_data3, offset_read_data_channel_3, out_read_channel_3, enable_read3, 3);
    join(H, W, I_ITER, out_read_channel_0, out_read_channel_1, out_read_channel_2, out_read_channel_3, out_read_data);
    read_bias(offset_bias, ptr_bias, out_read_bias);
    read_kernel(I_ITER, offset_kernel, ptr_kernel, out_read_kernel);
    conv(H, W, I_ITER, out_read_data, out_read_kernel, out_read_bias, out_conv);
    relu(enable_relu, H, W, out_conv, out_relu);
    split(H, W, out_relu, out_write_channel_0, out_write_channel_1, out_write_channel_2, out_write_channel_3);
    write_data_channel(H, W, ptr_out0, offset_write_data_channel_0, out_write_channel_0, enable_write0, 0);
    write_data_channel(H, W, ptr_out1, offset_write_data_channel_1, out_write_channel_1, enable_write1, 1);
    write_data_channel(H, W, ptr_out2, offset_write_data_channel_2, out_write_channel_2, enable_write2, 2);
    write_data_channel(H, W, ptr_out3, offset_write_data_channel_3, out_write_channel_3, enable_write3, 3);

  }

  #ifdef DEBUG_VERBOSE
  printf("kernel finishes\n");
  #endif
}

} // end extern "C"
