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
// This kernel computes different operations in a streaming line manner. The following operations are supported, in the same specific order they are listed here:
//   - scalmul. Multiplies the input data by a scalar value
//   - resize. Resizes the input data by a factor, leading to an upsize or downsize operation
//   - conv. Performs the convolution operation to the input data
//   - relu. Performs the ReLu activation operation (negative values are set to zero)
//   - maxpool. Performs the maxpooling operation on the input
//   - sigmoid. Performs the Sigmoid activation operation
//
// Each module can be disabled by an specific parameter of the kernel. Specifically, the following enable parameters are used:
//   enable_scalmul
//   enable_resize
//   enable_conv
//   enable_relu
//   enable_maxpool
//   enable_sigmoid
//
// All the operations are performed on a set on input channels and producing a set of output channels. These channels are known by CPI and CPO design constants.
// Parallel slicing is applied both at the input (CPI) and at the output (CPO) of the conv module. The rest of modules apply the same factor, those modules before
// the conv module use CPI and those after the conv module use CPO
//
// The pipeline flow is fed by three read modules from DDR memory. The read_bias module reads bias for the conv kernel. The read_kernel module reads the filters for 
// the convolution operation. These two modules will not read if the enable_conv parameter is FALSE. The read_kernel filter reads filters stored in memory in the
// following order: GO x GI x CPO x CPI x KH x KW. This ordering enables sequential reading from memory
// 
// read_data_channel module reads the data, expected to be in I x H x W format. CPI channels are read in chunks and sent through different output streams. Each stream
// reaches the S&F module where data is serialized and filtered in order to forward the correct number of elements for operation. All these items reach the join module
// where they are combined in sets of CPI pixels. These sets (pixel_in_t) are sent forward.
//
// Data is written into memory by the write_data_channel module. CPO channels are written in blocks in memory. For this, the split module separates pixels from different
// channels.
//
// The kernel works on iterations in order to accomplish a specific workload. The workload is defined as the image geometry (H and W) and the number of input (I) and 
// output (O) channels. If I > CPI then the module iterates on the input as many times as I / CPI. For the output it also iterates as many times as O / CPO. Two variables
// are used to control iterations: I_ITER and O_ITER (I_ITER = I / CPI, O_ITER = O / CPO).
//
// For the specific workload to compute, the module defines the number of pixels to process per channel, set into the variable num_pixels and passed as argument to the different
// modules.
//
// Modules:
//
//   scalmul. This module receives num_pixels sets of CPI pixels on each I_ITER iteration. On each cycle it receives CPI pixels, one from each channel. The sets of pixels
//            is forwarded with the pixels multiplied by the scalar argument. The module can be disabled by the enable_scalmul argument. In that case the input is forwarded
//            to the output with no modification of its contents.
//
//   resize.
//
//   conv.
//
//   relu.
//
//   maxpool.
//
//   sigmoid.
// The convolution operation can perform partial convolutions. The input image geometry (W, H) can be seen as a set of frames (W, rows) where num_fragments = H / rows
// The convolution can be instructed to perform the convolution of a frame. This enables overlapping different kernels on different frames at the same time.
//
// The kernel uses DataFlow model and is optimized in order to be bounded by the memory bandwidth.
//
//  Dataflow:
//
//   .......                                                                                                                                                                      .......
//   |     | ---> read_bias ----------------------------------------------------------------------                                                                                |     |
//   |     |                                                                                     |                                                                                |     |
//   |     | ---> read_kernel ------------------------------------------------------------       |                                                                                |     |
//   |     |     .....................    .....    .....                                 |       |                                             .....    ......................    |     |
//   |     |     |                   | -> |S&F| -> |   |                                 |       |                                             |   | -> |                    |    |     |
//   |     |     |                   |    .....    |   |    .........    ........    ................    ........    .........    .........    | s |    |                    |    |     |
//   |  D  |     |                   | -> |S&F| -> | j |    |       |    |      |    |     conv     |    |      |    |maxpool|    |       |    | p | -> |                    |    |  D  |
//   |  D  | --->| read_data_channel |    .....    | o | -> |scalmul| -> |resize| -> | K:3x3, P:1x1 | -> | relu | -> | K:2x2 | -> |sigmoid| -> | l |    | write_data_channel | -> |  D  |
//   |  R  |     |                   | -> |S&F| -> | i |    |       |    |      |    |     S:1x1    |    |      |    | S:2x2 |    |       |    | i | -> |                    |    |  R  |
//   |     |     |                   |    .....    | n |    .........    ........    ................    ........    .........    .........    | t |    |                    |    |     |
//   |     |     |                   | -> |S&F| -> |   |        |            |               |              |            |            |        |   | -> |                    |    |     |
//   |     |     .....................    .....    .....        |            |               |              |            |            |        .....    ......................    |     |
//   |     |                                                    |            |               |              |            |            |                                           |     |
//   |     |                                                 enable_      enable_         enable_        enable_      enable_      enable_                                        |     |
//   .......                                                 scalmul      resize          conv           relu         maxpool      sigmoid                                        .......
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
//    - rows: Number of rows of the frame to compute
//    - ptr_data: Memory pointer to input data
//    - ptr_kernel: Memory pointer to kernels
//    - ptr_bias: Memory pointer to bias
//    - ptr_out: Memory pointer to output buffer
//    - enable_relu: Enables RELU activation function at the end of the conv operation
//
//

// Headers
#include <math.h>
#include <stdio.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// Enable this define to get information (sw_emu)
//#define DEBUG_READ_BIAS
//#define DEBUG_READ_KERNEL
//#define DEBUG_READ_DATA
//#define DEBUG_DEMUX
//#define DEBUG_SERIALIZE
//#define DEBUG_JOIN
//#define DEBUG_PADDING
//#define DEBUG_SPLIT
//#define DEBUG_ALIGN
//#define DEBUG_WRITE_DATA

extern "C" {

// Data type to be used
//#define data_type float
#define data_type ap_fixed<8,4,AP_TRN,AP_WRAP>

// To allow using defines inside Xilinx pragmas
#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

// Fixed parameters (optimized at compilation/synthesis time)
#define KW       3  // kernel width
#define KH       3  // kernel height
#define CPI      8  // channels per input port
#define CPO      8  // channels per output port

// Maximum width and width*height
#define WMAX  256
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

// kernel struct
struct kernel_t {
  data_type pixel[CPO][CPI][9];
};

// blocks to read/write from/to memory
// Values optimized for ALVEO:
//     ap_fixed_8 bits : BLOCK_SIZE 64, CHUNK_SIZE 8
//     float           : BLOCK_SIZE 16, CHUNK_SIZE 64
//
#define BLOCK_SIZE 64            // block size is the number of pixels to read/write per cycle
#define CHUNK_SIZE 4             // chunk size is the number of consecutive blocks to read per channel before swithing to another channel
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

  #ifdef DEBUG_READ_BIAS
  printf("READ_BIAS: start\n");
  #endif

  pixel_out_t bias;
  #pragma HLS ARRAY_PARTITION variable=bias dim=0

  // we read the bias
  for (int i=0; i<CPO; i++) {
    data_type v = b_ptr[i + offset_bias];
    bias.pixel[i] = v;
  }

  #ifdef DEBUG_READ_BIAS
  printf("READ_BIAS: bias = ");
  for (int c=0; c<CPO; c++) printf(" %f ", float(bias.pixel[c]));
  printf("\n");
  #endif
  
  b_out << bias;

  #ifdef DEBUG_READ_BIAS
  printf("READ_BIAS: end\n");
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
static void read_kernel(int I_ITER, int offset_kernel, data_type *k_ptr, hls::stream<kernel_t> &k_out){

  #ifdef DEBUG_READ_KERNEL
  printf("READ_KERNEL: start\n");
  #endif

  // we read all the kernels and send them through the stream
  kernel_t kernel_k;
  #pragma HLS ARRAY_PARTITION variable=kernel_k dim=0
  int cpi = 0;
  int cpo = 0;
  int p = 0;

  int size = KW * KH * CPO * I_ITER * CPI;
  read_kernel_loop:
  for (int i=0; i<size; i++) {
    kernel_k.pixel[cpo][cpi][p] = k_ptr[i+ offset_kernel];
    p = p + 1;
    if (p == 9) {
      p = 0;
      cpi = cpi+1;
      if (cpi == CPI) {
        cpi = 0;
        cpo++;
        if (cpo == CPO) {
          cpo = 0;
	        k_out << kernel_k;
          #ifdef DEBUG_READ_KERNEL
	        printf("READ_KERNEL: kernel read\n");
	        for (int co=0; co<CPO; co++) {
	          for (int c=0; c<CPI; c++) {
              printf("channel co %d ci %d: ", co, c);
	            for (int p=0; p<9; p++) printf(" %f ", float(kernel_k.pixel[co][c][p]));
	            printf("\n");
            }
	        }
          #endif
        }
      }
    }
  }

  #ifdef DEBUG_READ_KERNEL
  printf("READ_KERNEL: end\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// read_data_channel. Reads one data channel and sends it through the stream
//
// Arguments:
//   H, W                : Data channel height and width
//   rows                : Number of rows of the frame to read
//   num_extra_rows      : Number of extra rows to read
//   I_ITER              : Number of input iterations (I / CPI)
//   ptr                 : pointer to input data
//   offset              : offsets within input data for each channel
//   out                 : output streams for each channel
//   enable              : enables for each channel. If not set the module produces just zeros and does not read memory
//
// If I_ITER > 1 the module reads several input channels. An stride between read channels
// is computed. 
//
static void read_data_channels(int H, int W, int rows, int I_ITER, block_t *ptr, int offset,
                                                                                 int num_extra_rows,
                                                                                 int channel_blocks, 
                                                                                 hls::stream<block_t> &out0, 
                                                                                 hls::stream<block_t> &out1, 
                                                                                 hls::stream<block_t> &out2, 
                                                                                 hls::stream<block_t> &out3, 
                                                                                 hls::stream<block_t> &out4, 
                                                                                 hls::stream<block_t> &out5, 
                                                                                 hls::stream<block_t> &out6, 
                                                                                 hls::stream<block_t> &out7, 
                                                                                 int enable0, int enable1, int enable2, int enable3,
                                                                                 int enable4, int enable5, int enable6, int enable7) {

  int num_pixels = (num_extra_rows + rows) * W;
  int channel_size = H * W;

  #ifdef DEBUG_READ_DATA
  printf("DEBUG: Read_data_channels starts\n");
  #endif

  for (int i_iter = 0; i_iter < I_ITER; i_iter++) {                                                                                  

    // each channel has its first block
    int offset_0 = offset + (channel_size * CPI * i_iter);
    int offset_1 = offset + (channel_size * CPI * i_iter) + (channel_size);
    int offset_2 = offset + (channel_size * CPI * i_iter) + (channel_size * 2);
    int offset_3 = offset + (channel_size * CPI * i_iter) + (channel_size * 3);
    int offset_4 = offset + (channel_size * CPI * i_iter) + (channel_size * 4);
    int offset_5 = offset + (channel_size * CPI * i_iter) + (channel_size * 5);
    int offset_6 = offset + (channel_size * CPI * i_iter) + (channel_size * 6);
    int offset_7 = offset + (channel_size * CPI * i_iter) + (channel_size * 7);
    int first_block_0 = offset_0 / BLOCK_SIZE;
    int first_block_1 = offset_1 / BLOCK_SIZE;
    int first_block_2 = offset_2 / BLOCK_SIZE;
    int first_block_3 = offset_3 / BLOCK_SIZE;
    int first_block_4 = offset_4 / BLOCK_SIZE;
    int first_block_5 = offset_5 / BLOCK_SIZE;
    int first_block_6 = offset_6 / BLOCK_SIZE;
    int first_block_7 = offset_7 / BLOCK_SIZE;

    // We read in chunks of CHUNK_SIZE blocks
    int num_chunks = (channel_blocks + CHUNK_SIZE - 1) / CHUNK_SIZE;

    int channel_blocks_remaining_0 = channel_blocks;
    int channel_blocks_remaining_1 = channel_blocks;
    int channel_blocks_remaining_2 = channel_blocks;
    int channel_blocks_remaining_3 = channel_blocks;
    int channel_blocks_remaining_4 = channel_blocks;
    int channel_blocks_remaining_5 = channel_blocks;
    int channel_blocks_remaining_6 = channel_blocks;
    int channel_blocks_remaining_7 = channel_blocks;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
      // Channel 0 chunk
      if (enable0) {
        for (int b=0; b<CHUNK_SIZE; b++) {
          #pragma HLS pipeline
          block_t bx;
          bx = ptr[first_block_0 + b];
          if (channel_blocks_remaining_0) out0 << bx;
          if (channel_blocks_remaining_0) channel_blocks_remaining_0 = channel_blocks_remaining_0 - 1;
        }
      }  
      first_block_0 = first_block_0 + CHUNK_SIZE;
      // Channel 1 chunk
      if (enable1) {
        for (int b=0; b<CHUNK_SIZE; b++) {
          #pragma HLS pipeline
          block_t bx;
          bx = ptr[first_block_1 + b];
          if (channel_blocks_remaining_1) out1 << bx;
          if (channel_blocks_remaining_1) channel_blocks_remaining_1 = channel_blocks_remaining_1 - 1;
        }
      } 
      first_block_1 = first_block_1 + CHUNK_SIZE;
      // Channel 2 chunk
      if (enable2) {
        for (int b=0; b<CHUNK_SIZE; b++) {
          #pragma HLS pipeline
          block_t bx;
          bx = ptr[first_block_2 + b];
          if (channel_blocks_remaining_2) out2 << bx;
          if (channel_blocks_remaining_2) channel_blocks_remaining_2 = channel_blocks_remaining_2 - 1;
        }
      }
      first_block_2 = first_block_2 + CHUNK_SIZE;
      // Channel 3 chunk
      if (enable3) {
        for (int b=0; b<CHUNK_SIZE; b++) {
          #pragma HLS pipeline
          block_t bx;
          bx = ptr[first_block_3 + b];
          if (channel_blocks_remaining_3) out3 << bx;
          if (channel_blocks_remaining_3) channel_blocks_remaining_3 = channel_blocks_remaining_3 - 1;
        }
      } 
      first_block_3 = first_block_3 + CHUNK_SIZE;
      // Channel 4 chunk
      if (enable4) {
        for (int b=0; b<CHUNK_SIZE; b++) {
          #pragma HLS pipeline
          block_t bx;
          bx = ptr[first_block_4 + b];
          if (channel_blocks_remaining_4) out4 << bx;
          if (channel_blocks_remaining_4) channel_blocks_remaining_4 = channel_blocks_remaining_4 - 1;
        }
      }  
      first_block_4 = first_block_4 + CHUNK_SIZE;
      // Channel 5 chunk
      if (enable5) {
        for (int b=0; b<CHUNK_SIZE; b++) {
          #pragma HLS pipeline
          block_t bx;
          bx = ptr[first_block_5 + b];
          if (channel_blocks_remaining_5) out5 << bx;
          if (channel_blocks_remaining_5) channel_blocks_remaining_5 = channel_blocks_remaining_5 - 1;
        }
      } 
      first_block_5 = first_block_5 + CHUNK_SIZE;
      // Channel 6 chunk
      if (enable6) {
        for (int b=0; b<CHUNK_SIZE; b++) {
          #pragma HLS pipeline
          block_t bx;
          bx = ptr[first_block_6 + b];
          if (channel_blocks_remaining_6) out6 << bx;
          if (channel_blocks_remaining_6) channel_blocks_remaining_6 = channel_blocks_remaining_6 - 1;
        }
      }
      first_block_6 = first_block_6 + CHUNK_SIZE;
      // Channel 7 chunk
      if (enable7) {
        for (int b=0; b<CHUNK_SIZE; b++) {
          #pragma HLS pipeline
          block_t bx;
          bx = ptr[first_block_7 + b];
          if (channel_blocks_remaining_7) out7 << bx;
          if (channel_blocks_remaining_7) channel_blocks_remaining_7 = channel_blocks_remaining_7 - 1;
        }
      } 
      first_block_7 = first_block_7 + CHUNK_SIZE;
    }
  }

  #ifdef DEBUG_READ_DATA
  printf("DEBUG: Read_data_channels ends\n");
  #endif
}

// ------------------
void serialize_and_filter(int I_ITER, int num_pixels, int channel_blocks, int channel_size, int offset, hls::stream<block_t> &in, hls::stream<data_type> &out, int enable) {

  #ifdef DEBUG_SERIALIZE
  printf("SERIALIZE: starts (num_pixels = %d)\n", num_pixels);
  #endif

  int num_pixels_cnt;

  // Zero block initialization
  block_t data_zeros;
  for (int b=0; b<BLOCK_SIZE; b++) {
    #pragma HLS UNROLL
    data_zeros.pixel[b] = 0;
  }

  int iters = I_ITER * channel_blocks * BLOCK_SIZE;

  int b = 0;
  int p = 0;
  int iter = 0;
  for (int i_iter=0; i_iter < iters; i_iter++) {
    #pragma HLS pipeline II=1
    // offset
    int offset_ch;
    if ((b==0) && (p==0)) {
      offset_ch = (offset + (channel_size * CPI * iter)) % BLOCK_SIZE;
      num_pixels_cnt = num_pixels;
    }
    block_t bx;
    DO_PRAGMA(HLS ARRAY_PARTITION variable=bx complete)
    if (p==0) {
      if (enable) bx = in.read(); else bx = data_zeros; 
    }
    if ((offset_ch==0) && (num_pixels_cnt !=0)) {
      out << bx.pixel[p];
      num_pixels_cnt = num_pixels_cnt - 1;
      #ifdef DEBUG_SERIALIZE
      printf("SERIALIZE: pixel forwarded %f\n", (float)bx.pixel[p]);
      #endif
    } else {
      offset_ch = offset_ch - 1;
    }
    p = p + 1;
    if (p == BLOCK_SIZE) {
      p = 0;
      b = b + 1;
      if (b == channel_blocks) {
        b = 0;
        iter = iter + 1;
      }
    }
  }
  
  #ifdef DEBUG_SERIALIZE
  printf("SERIALIZE: ends (remaining pixels to send %d)\n", num_pixels_cnt);
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
static void join(int H, int W, int I_ITER, int num_extra_rows,
                               hls::stream<data_type> &in0, hls::stream<data_type> &in1, hls::stream<data_type> &in2, hls::stream<data_type> &in3, 
                               hls::stream<data_type> &in4, hls::stream<data_type> &in5, hls::stream<data_type> &in6, hls::stream<data_type> &in7, 
                               hls::stream<pixel_in_t> &out) {

  #ifdef DEBUG_JOIN
  printf("JOIN: starts\n");
  #endif

  int num_pixels = (H + num_extra_rows) * W;                    // pixels to read
   
  #ifdef DEBUG_JOIN
  printf("JOIN: Expected pixels = %d\n", num_pixels);
  #endif

  for (int i_iter = 0; i_iter < I_ITER; i_iter++) {

    join_loop:
    for (int r=0; r<num_pixels; r++) {
      #pragma HLS PIPELINE II=1
      pixel_in_t data;
      data.pixel[0] = in0.read();
      data.pixel[1] = in1.read();
      data.pixel[2] = in2.read();
      data.pixel[3] = in3.read();
      data.pixel[4] = in4.read();
      data.pixel[5] = in5.read();
      data.pixel[6] = in6.read();
      data.pixel[7] = in7.read();
      out << data;
    }
  }
  #ifdef DEBUG_JOIN
  printf("JOIN: ends\n");
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
static void split(int H, int W, hls::stream<pixel_out_t> &in, hls::stream<block_t> &out0, hls::stream<block_t> &out1, hls::stream<block_t> &out2, hls::stream<block_t> &out3,
                                                              hls::stream<block_t> &out4, hls::stream<block_t> &out5, hls::stream<block_t> &out6, hls::stream<block_t> &out7) {

  #ifdef DEBUG_SPLIT
  printf("SPLIT: starts (num pixels %d)\n", H * W);
  #endif

  int num_pixels = H * W;                                       // pixels to receive per channel
  int b = 0;
  block_t b0;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b0 complete)
  block_t b1;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b1 complete)
  block_t b2;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b2 complete)
  block_t b3;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b3 complete)
  block_t b4;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b4 complete)
  block_t b5;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b5 complete)
  block_t b6;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b6 complete)
  block_t b7;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b7 complete)

  split_loop:
  for (int r=0; r<num_pixels; r++) {
    #pragma HLS PIPELINE II=1
    pixel_out_t data;
    data = in.read();
    #ifdef DEBUG_SPLIT
    printf("SPLIT: splits eight pixels %f %f %f %f %f %f %f %f on iteration %d\n", float(data.pixel[0]), float(data.pixel[1]), float(data.pixel[2]), float(data.pixel[3]), float(data.pixel[4]), float(data.pixel[5]), float(data.pixel[6]), float(data.pixel[7]), r);
    #endif
    b0.pixel[b] = data.pixel[0];
    b1.pixel[b] = data.pixel[1];
    b2.pixel[b] = data.pixel[2];
    b3.pixel[b] = data.pixel[3];
    b4.pixel[b] = data.pixel[4];
    b5.pixel[b] = data.pixel[5];
    b6.pixel[b] = data.pixel[6];
    b7.pixel[b] = data.pixel[7];
    b = b + 1;
    if (b == BLOCK_SIZE) {
      out0 << b0;
      out1 << b1;
      out2 << b2;
      out3 << b3;
      out4 << b4;
      out5 << b5;
      out6 << b6;
      out7 << b7;
      b = 0;
    }
  }

  #ifdef DEBUG_SPLIT
  printf("SPLIT: ends\n");
  #endif
}

// ---------------------------------------------------------------------------------------
// write_data_channels. Writes data channels from the elements read from an input stream
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
static void write_data_channels(int num_pixels, block_t *ptr, int offset0, hls::stream<block_t> &in0, int enable0,
                                                                int offset1, hls::stream<block_t> &in1, int enable1,
                                                                int offset2, hls::stream<block_t> &in2, int enable2,
                                                                int offset3, hls::stream<block_t> &in3, int enable3,
                                                                int offset4, hls::stream<block_t> &in4, int enable4,
                                                                int offset5, hls::stream<block_t> &in5, int enable5,
                                                                int offset6, hls::stream<block_t> &in6, int enable6,
                                                                int offset7, hls::stream<block_t> &in7, int enable7) {
  #ifdef DEBUG_WRITE_DATA
  printf("WRITE_DATA: starts\n");
  #endif

  int offset0_write = offset0 / BLOCK_SIZE;
  int offset1_write = offset1 / BLOCK_SIZE;
  int offset2_write = offset2 / BLOCK_SIZE;
  int offset3_write = offset3 / BLOCK_SIZE;
  int offset4_write = offset4 / BLOCK_SIZE;
  int offset5_write = offset5 / BLOCK_SIZE;
  int offset6_write = offset6 / BLOCK_SIZE;
  int offset7_write = offset7 / BLOCK_SIZE;

  int num_blocks = num_pixels / BLOCK_SIZE;

  int num_channels = 4;
  int offset;

  #ifdef DEBUG_WRITE_DATA
  printf("WRITE_DATA: pixels  = %d\n", num_pixels);
  printf("WRITE_DATA: enables = %d %d %d %d %d %d %d %d\n", enable0, enable1, enable2, enable3, enable4, enable5, enable6, enable7);
  printf("WRITE_DATA: Offsets = %d %d %d %d %d %d %d %d\n", offset0, offset1, offset2, offset3, offset4, offset5, offset6, offset7);
  #endif

  int channel = 0;
  int pos = 0;
  int slot = 0;
  block_t bx;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=bx complete)

  for (int p = 0; p < num_blocks * CPO; p++) {
    #pragma HLS pipeline II=1
    if (channel == 0) bx = in0.read(); else if (channel == 1) bx = in1.read(); else if (channel == 2) bx = in2.read(); else if (channel ==3) bx = in3.read(); else if (channel == 4) bx = in4.read(); else if (channel == 5) bx = in5.read(); else if (channel == 6) bx = in6.read(); else bx = in7.read();
    int enable   = (channel == 0) ? enable0    : (channel==1) ? enable1    : (channel==2) ? enable2    : (channel == 3) ? enable3    : (channel==4) ? enable4    : (channel==5) ? enable5    : (channel == 6) ? enable6 : enable7;
    int offset   = (channel == 0) ? offset0_write : (channel==1) ? offset1_write    : (channel==2) ? offset2_write    : (channel==3) ? offset3_write : (channel==4) ? offset4_write : (channel==5) ? offset5_write : (channel==6) ? offset6_write : offset7_write;
    if (enable) {ptr[offset] = bx;}
    if ((enable0) && (channel == 0)) offset0_write = offset0_write + 1;
    if ((enable1) && (channel == 1)) offset1_write = offset1_write + 1;
    if ((enable2) && (channel == 2)) offset2_write = offset2_write + 1;
    if ((enable3) && (channel == 3)) offset3_write = offset3_write + 1;
    if ((enable4) && (channel == 4)) offset4_write = offset4_write + 1;
    if ((enable5) && (channel == 5)) offset5_write = offset5_write + 1;
    if ((enable6) && (channel == 6)) offset6_write = offset6_write + 1;
    if ((enable7) && (channel == 7)) offset7_write = offset7_write + 1;
    channel = (channel + 1) % CPO;
  }
  #ifdef DEBUG_WRITE_DATA
  printf("WRITE_DATA: ends\n");
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
static void padding(int H, int W, int I_ITER, int enable_upper_padding, int enable_lower_padding, hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out) {

  #ifdef DEBUG_PADDING
  printf("PADDING: start\n");
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
        if (((enable_upper_padding==1) && (h==0)) || ((enable_lower_padding==1) && (h == H+1)) || w == 0 || w == W+1) {
          data = zero;
        } else {
          data = in.read();
        }
        #ifdef DEBUG_PADDING
        for(int cpi = 0;cpi<CPI;cpi++) printf("PADDING: data.pixel[%d] = %6.2f  ", cpi, float(data.pixel[cpi]));
        printf("\n");
        #endif
        out << data;
      }
    }
  } // iter

  #ifdef DEBUG_PADDING
  printf("PADDING: end\n");
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
static void mul(int H, int W, int I_ITER, hls::stream<frame_t> &in, hls::stream<kernel_t> &k_in, hls::stream<pixel_out_t> &out) {

  #ifdef DEBUG_VERBOSE
  printf("mul: start\n");
  #endif

  kernel_t kernel;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=kernel dim=0)
  frame_t data_in;

  data_type sum[CPO];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=sum dim=0 block factor=CPO)

  pixel_out_t p_out;

  int load_kernel = 0;
  int num_iter = I_ITER * H * W;
  int iter_load_kernel = 0;

  //
  mul_loop:
  for(int i = 0; i < num_iter; i++){
    #pragma HLS PIPELINE II=1
    load_kernel = (iter_load_kernel == 0);
    if (load_kernel) kernel = k_in.read();
    sum[0] = 0;
    sum[1] = 0;
    sum[2] = 0;
    sum[3] = 0;
    sum[4] = 0;
    sum[5] = 0;
    sum[6] = 0;
    sum[7] = 0;
  
    data_in = in.read();

    loop_mul_cpi:
    for (int cpi=0; cpi<CPI; cpi++) {
      #pragma HLS UNROLL
      loop_mul_j:
      for (int j=0; j<KW*KH; j++) {
        #pragma HLS UNROLL
        loop_mul_cpo:
        for (int cpo=0; cpo<CPO; cpo++) {
          #pragma HLS UNROLL
          sum[cpo] += data_in.pixel[j].pixel[cpi] * kernel.pixel[cpo][cpi][j];
        }
      }
    }

    p_out.pixel[0] = sum[0];
    p_out.pixel[1] = sum[1];
    p_out.pixel[2] = sum[2];
    p_out.pixel[3] = sum[3];
    p_out.pixel[4] = sum[4];
    p_out.pixel[5] = sum[5];
    p_out.pixel[6] = sum[6];
    p_out.pixel[7] = sum[7];
    out << p_out;
    iter_load_kernel++;
    if (iter_load_kernel == W*H) iter_load_kernel = 0;
  }

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
static void conv(int H, int W, int I_ITER, int enable_upper_padding, int enable_lower_padding, hls::stream<pixel_in_t> &in, hls::stream<kernel_t> &k_in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

  // streams
  static hls::stream<pixel_in_t>  str_pad_cvt;  // padding->cvt
  static hls::stream<frame_t>     str_cvt_mul;  // cvt->mul
  static hls::stream<pixel_out_t> str_mul_add;  // mul->add

  // topology
  #pragma HLS dataflow
  padding(H, W, I_ITER, enable_upper_padding, enable_lower_padding, in, str_pad_cvt);            // padding
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
void k_conv2D_8x8
                                (
		                             block_t *ptr_data,
                                 int H, int W, int rows, int I, int O, int I_ITER, int O_ITER, int enable_relu,
                                 data_type *ptr_kernel, data_type *ptr_bias,
                                 block_t *ptr_out, 
                                 int global_offset, int enable_upper_padding, int enable_lower_padding
				) {

  #pragma HLS INTERFACE s_axilite port=W bundle=control
  #pragma HLS INTERFACE s_axilite port=H bundle=control
  #pragma HLS INTERFACE s_axilite port=rows bundle=control
  #pragma HLS INTERFACE s_axilite port=I bundle=control
  #pragma HLS INTERFACE s_axilite port=O bundle=control
  #pragma HLS INTERFACE s_axilite port=I_ITER bundle=control
  #pragma HLS INTERFACE s_axilite port=O_ITER bundle=control
  #pragma HLS INTERFACE s_axilite port=enable_relu bundle=control
  #pragma HLS INTERFACE s_axilite port=global_offset bundle=control
  #pragma HLS INTERFACE s_axilite port=enable_upper_padding bundle=control
  #pragma HLS INTERFACE s_axilite port=enable_lower_padding bundle=control
  #pragma HLS INTERFACE m_axi port=ptr_data offset=slave bundle=gmem   
  #pragma HLS INTERFACE m_axi port=ptr_kernel offset=slave bundle=gmem1 
  #pragma HLS INTERFACE m_axi port=ptr_bias offset=slave bundle=gmem2 
  #pragma HLS INTERFACE m_axi port=ptr_out offset=slave bundle=gmem  
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  // ptr_data struct to be packed as a single element vector (to improve memory read)
  // the compiler will do full structure access (all elements of structure)
  #pragma HLS data_pack variable = ptr_data
  #pragma HLS data_pack variable = ptr_out

  #ifdef DEBUG_VERBOSE
  printf("kernel starts...\n");
  #endif

  for (int o_iter = 0; o_iter<O_ITER; o_iter++) {

    #pragma HLS dataflow

    // we compute the enable_write signals
    int o_channel = (o_iter << 3);  // current output channel (first one in this iteration)
    int enable_write0 = o_channel < O;
    int enable_write1 = (o_channel + 1) < O;
    int enable_write2 = (o_channel + 2) < O;
    int enable_write3 = (o_channel + 3) < O;
    int enable_write4 = (o_channel + 4) < O;
    int enable_write5 = (o_channel + 5) < O;
    int enable_write6 = (o_channel + 6) < O;
    int enable_write7 = (o_channel + 7) < O;

    // we compute the enable_read signals
    int enable_read0 = (I >= 1);
    int enable_read1 = (I >= 2);
    int enable_read2 = (I >= 3);
    int enable_read3 = (I >= 4);
    int enable_read4 = (I >= 5);
    int enable_read5 = (I >= 6);
    int enable_read6 = (I >= 7);
    int enable_read7 = (I >= 8);

    // input and output streams
    static hls::stream<pixel_in_t> out_read_data;
    static hls::stream<kernel_t> out_read_kernel;
    static hls::stream<pixel_out_t> out_read_bias;
    static hls::stream<pixel_out_t> out_conv;
    //ReLu stream
    static hls::stream<pixel_out_t> out_relu;

    //static hls::stream<block_t>   stream_data_read;

    //#pragma HLS stream variable=stream_data_read depth=64

    static hls::stream<block_t>   stream_data_ch0_0;
    static hls::stream<data_type> stream_data_ch0_1;
    static hls::stream<block_t>   stream_data_ch1_0;
    static hls::stream<data_type> stream_data_ch1_1;
    static hls::stream<block_t>   stream_data_ch2_0;
    static hls::stream<data_type> stream_data_ch2_1;
    static hls::stream<block_t>   stream_data_ch3_0;
    static hls::stream<data_type> stream_data_ch3_1;
    static hls::stream<block_t>   stream_data_ch4_0;
    static hls::stream<data_type> stream_data_ch4_1;
    static hls::stream<block_t>   stream_data_ch5_0;
    static hls::stream<data_type> stream_data_ch5_1;
    static hls::stream<block_t>   stream_data_ch6_0;
    static hls::stream<data_type> stream_data_ch6_1;
    static hls::stream<block_t>   stream_data_ch7_0;
    static hls::stream<data_type> stream_data_ch7_1;

    DO_PRAGMA(HLS stream variable=stream_data_ch0_0 depth=CHUNK_SIZE)
    DO_PRAGMA(HLS stream variable=stream_data_ch1_0 depth=CHUNK_SIZE)
    DO_PRAGMA(HLS stream variable=stream_data_ch2_0 depth=CHUNK_SIZE)
    DO_PRAGMA(HLS stream variable=stream_data_ch3_0 depth=CHUNK_SIZE)
    DO_PRAGMA(HLS stream variable=stream_data_ch4_0 depth=CHUNK_SIZE)
    DO_PRAGMA(HLS stream variable=stream_data_ch5_0 depth=CHUNK_SIZE)
    DO_PRAGMA(HLS stream variable=stream_data_ch6_0 depth=CHUNK_SIZE)
    DO_PRAGMA(HLS stream variable=stream_data_ch7_0 depth=CHUNK_SIZE)

    static hls::stream<block_t> out_write_channel_0;
    static hls::stream<block_t> out_write_channel_1;
    static hls::stream<block_t> out_write_channel_2;
    static hls::stream<block_t> out_write_channel_3;
    static hls::stream<block_t> out_write_channel_4;
    static hls::stream<block_t> out_write_channel_5;
    static hls::stream<block_t> out_write_channel_6;
    static hls::stream<block_t> out_write_channel_7;

    // channel offsets for reading
    int corrected_offset = (enable_upper_padding==0)? W : 0;
    int channel_offset = (W * H);
    int num_extra_rows = (enable_lower_padding == 0) + (enable_upper_padding == 0);
    int offset_read_data_channel = global_offset - corrected_offset;

    int channel_size = H * W;
    int read_pixels = W * (rows + num_extra_rows);
    int write_pixels = rows * W;
    int channel_blocks = (read_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("channel_size %d, read_pixels %d, write_pixels %d, channel_blocks = %d\n", channel_size, read_pixels, write_pixels, channel_blocks);

    // channel offsets for reading
    int offset_read_data_channel_0 = offset_read_data_channel % BLOCK_SIZE;
    int offset_read_data_channel_1 = (offset_read_data_channel + channel_offset) % BLOCK_SIZE;
    int offset_read_data_channel_2 = (offset_read_data_channel + 2 * channel_offset) % BLOCK_SIZE;
    int offset_read_data_channel_3 = (offset_read_data_channel + 3 * channel_offset) % BLOCK_SIZE;
    int offset_read_data_channel_4 = (offset_read_data_channel + 4 * channel_offset) % BLOCK_SIZE;
    int offset_read_data_channel_5 = (offset_read_data_channel + 5 * channel_offset) % BLOCK_SIZE;
    int offset_read_data_channel_6 = (offset_read_data_channel + 6 * channel_offset) % BLOCK_SIZE;
    int offset_read_data_channel_7 = (offset_read_data_channel + 7 * channel_offset) % BLOCK_SIZE;
    
    // channel offsets for writing
    int offset_write_data_channel_0 = global_offset + ((o_iter << 3) * channel_offset);
    int offset_write_data_channel_1 = global_offset + channel_offset + ((o_iter << 3) * channel_offset);
    int offset_write_data_channel_2 = global_offset + (channel_offset * 2) + ((o_iter << 3) * channel_offset);
    int offset_write_data_channel_3 = global_offset + (channel_offset * 3) + ((o_iter << 3) * channel_offset);
    int offset_write_data_channel_4 = global_offset + (channel_offset * 4) + ((o_iter << 3) * channel_offset);
    int offset_write_data_channel_5 = global_offset + (channel_offset * 5) + ((o_iter << 3) * channel_offset);
    int offset_write_data_channel_6 = global_offset + (channel_offset * 6) + ((o_iter << 3) * channel_offset);
    int offset_write_data_channel_7 = global_offset + (channel_offset * 7) + ((o_iter << 3) * channel_offset);

    int offset_bias = o_iter << 3;
    int offset_kernel = o_iter * (I < 8 ? 8 : I) * 9 * CPO;
                                                                                   
    read_data_channels(H, W, rows, I_ITER, ptr_data, offset_read_data_channel, num_extra_rows, channel_blocks, stream_data_ch0_0, stream_data_ch1_0, stream_data_ch2_0, stream_data_ch3_0, 
                                                                                                               stream_data_ch4_0, stream_data_ch5_0, stream_data_ch6_0, stream_data_ch7_0,
                                                                                                               enable_read0, enable_read1, enable_read2, enable_read3,
                                                                                                               enable_read4, enable_read5, enable_read6, enable_read7);
    serialize_and_filter(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_0, stream_data_ch0_0, stream_data_ch0_1, enable_read0);
    serialize_and_filter(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_1, stream_data_ch1_0, stream_data_ch1_1, enable_read1);
    serialize_and_filter(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_2, stream_data_ch2_0, stream_data_ch2_1, enable_read2);
    serialize_and_filter(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_3, stream_data_ch3_0, stream_data_ch3_1, enable_read3);
    serialize_and_filter(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_4, stream_data_ch4_0, stream_data_ch4_1, enable_read4);
    serialize_and_filter(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_5, stream_data_ch5_0, stream_data_ch5_1, enable_read5);
    serialize_and_filter(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_6, stream_data_ch6_0, stream_data_ch6_1, enable_read6);
    serialize_and_filter(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_7, stream_data_ch7_0, stream_data_ch7_1, enable_read7);
    join(rows, W, I_ITER, num_extra_rows, stream_data_ch0_1, stream_data_ch1_1, stream_data_ch2_1, stream_data_ch3_1, 
                                          stream_data_ch4_1, stream_data_ch5_1, stream_data_ch6_1, stream_data_ch7_1, out_read_data);
    read_bias(offset_bias, ptr_bias, out_read_bias);
    read_kernel(I_ITER, offset_kernel, ptr_kernel, out_read_kernel);
    conv(rows, W, I_ITER, enable_upper_padding, enable_lower_padding, out_read_data, out_read_kernel, out_read_bias, out_conv);
    relu(enable_relu, rows, W, out_conv, out_relu);
    split(rows, W, out_relu, out_write_channel_0, out_write_channel_1, out_write_channel_2, out_write_channel_3,
                             out_write_channel_4, out_write_channel_5, out_write_channel_6, out_write_channel_7);
    write_data_channels(write_pixels, ptr_out, offset_write_data_channel_0, out_write_channel_0, enable_write0,
                                          offset_write_data_channel_1, out_write_channel_1, enable_write1,
                                          offset_write_data_channel_2, out_write_channel_2, enable_write2,
                                          offset_write_data_channel_3, out_write_channel_3, enable_write3,
                                          offset_write_data_channel_4, out_write_channel_4, enable_write4,
                                          offset_write_data_channel_5, out_write_channel_5, enable_write5,
                                          offset_write_data_channel_6, out_write_channel_6, enable_write6,
                                          offset_write_data_channel_7, out_write_channel_7, enable_write7);
  }

  #ifdef DEBUG_VERBOSE
  printf("kernel finishes\n");
  #endif
}

} // end extern "C"
