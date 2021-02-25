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
#include <ap_int.h>
#include <hls_stream.h>

// Enable this define to get information (sw_emu)
// #define DEBUG_VERBOSE
// #define DEBUG_READ_BIAS
// #define DEBUG_READ_KERNEL
// #define DEBUG_READ_DATA
// #define DEBUG_SERIALIZE
// #define DEBUG_JOIN
// #define DEBUG_SPLIT
// #define DEBUG_WRITE_DATA
// #define DEBUG_RELU
// #define DEBUG_PADDING
// #define DEBUG_CVT
// #define DEBUG_MUL
// #define DEBUG_ADD

// Data type to be used
// #define data_type float
// #define data_type ap_fixed<8,4,AP_TRN,AP_WRAP>
#define data_type ap_int<8>

// To allow using defines inside Xilinx pragmas
#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

// Fixed parameters (optimized at compilation/synthesis time)
#define KW       3  // kernel width
#define KH       3  // kernel height
#define CPI      4  // channels per input port
#define CPO      4  // channels per output port

// Maximum width and width*height
#define HMAX  256
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

//kernel read struct
struct kernel_in_t {
  data_type pixel[9];
};

// blocks to read/write from/to memory
// Values optimized for ALVEO:
//     ap_fixed_8 bits : BLOCK_SIZE 64, CHUNK_SIZE 8
//     float           : BLOCK_SIZE 16, CHUNK_SIZE 64
//
#define BLOCK_SIZE 64            // block size is the number of pixels to read/write per cycle
#define CHUNK_SIZE 8            // chunk size is the number of consecutive blocks to read per channel before swithing to another channel
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
static void read_bias(int offset_bias, pixel_out_t *b_ptr, hls::stream<pixel_out_t> &b_out) {

  #ifdef DEBUG_READ_BIAS
  printf("READ_BIAS: start\n");
  #endif

  pixel_out_t bias;
  #pragma HLS ARRAY_PARTITION variable=bias complete dim=0

  bias = b_ptr[offset_bias];
  #ifdef DEBUG_READ_BIAS
  printf("READ_BIAS: offset_bias = %d", offset_bias);
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
static void read_kernel(int I_ITER, int offset_kernel, kernel_in_t *k_ptr, hls::stream<kernel_t> &k_out){

  #ifdef DEBUG_READ_KERNEL
  printf("READ_KERNEL: start\n");
  #endif

  // we read all the kernels and send them through the stream
  kernel_t kernel_k;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=kernel_k complete dim=3)

  int num_kernels = 0;
  kernel_in_t k_in;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=k_in dim=0 complete)
  i_iter_read_kernel:
  for (int i_iter=0; i_iter<I_ITER; i_iter++){
    cpi_read_kernel:
    for (int i=0; i<CPO; i++) {
      cpo_read_kernel:
      for (int o=0; o<CPI; o++){
        #pragma HLS pipeline II=1
        k_in = k_ptr[num_kernels +  offset_kernel];
        num_kernels ++;
        #ifdef DEBUG_READ_KERNEL
        printf("READ_KERNEL: offset_kernel = %d \n", offset_kernel);
        printf("READ_KERNEL: iteracion = %d \n", num_kernels);
        printf("READ_KERNEL: kernel read\n");
            for (int p=0; p<9; p++){
              printf(" %f ", float(k_in.pixel[p]));
              if((p+1)%3==0)printf("\n");
            }
            printf("\n");
        #endif
        p_read_kernel:
        for (int p=0; p<9; p++){
          #pragma HLS UNROLL
          kernel_k.pixel[i][o][p] = k_in.pixel[p];
        }
      }
    }
    k_out << kernel_k;
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
static void read_data_channels(int H, int W, int rows, int I_ITER, block_t *ptr, int offset, int num_extra_rows, int channel_blocks, hls::stream<block_t> out[CPI], int *enable_read) {

  //original
  int num_pixels = (num_extra_rows + rows) * W;
  int channel_size = H * W;

  #ifdef DEBUG_READ_DATA
  printf("DEBUG: Read_data_channels starts\n");
  #endif

  for (int i_iter = 0; i_iter < I_ITER; i_iter++) {

    // each channel has its first block
    int offset_[CPI];
    int first_block_[CPI];
    // We read in chunks of CHUNK_SIZE blocks
    int num_chunks = (channel_blocks + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int channel_blocks_remaining_[CPI];
    for(int i = 0; i<CPI; i++){
      #pragma HLS UNROLL
      offset_[i] = offset + (channel_size * CPI * i_iter) + (channel_size * i);
      first_block_[i] = offset_[i] / BLOCK_SIZE;
      channel_blocks_remaining_[i] = channel_blocks;
    }

    for (int chunk = 0; chunk < num_chunks; chunk++) {
      for(int i = 0; i < CPI; i++){
        #pragma HLS UNROLL
        // Channel i chunk
        if (enable_read[i]) {
          for (int b=0; b<CHUNK_SIZE; b++) {
            #pragma HLS pipeline
            block_t bx;
            bx = ptr[first_block_[i] + b];
            if (channel_blocks_remaining_[i]) out[i] << bx;
            if (channel_blocks_remaining_[i]) channel_blocks_remaining_[i] = channel_blocks_remaining_[i] - 1;
          }
        }
        first_block_[i] = first_block_[i] + CHUNK_SIZE;

      }
    }
  } //i_iter

  #ifdef DEBUG_READ_DATA
  printf("DEBUG: Read_data_channels ends\n");
  #endif
}

// ------------------
static void serialize_and_filter(int I_ITER, int num_pixels, int channel_blocks, int channel_size, int offset, hls::stream<block_t> &in, hls::stream<data_type> &out, int enable) {

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
    DO_PRAGMA(HLS ARRAY_PARTITION variable=bx dim=0 complete)
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

template <int LEVELS>
static void ch_serialize_and_filter(int I_ITER, int num_pixels, int channel_blocks, int channel_size,
                                                                int *offset_read_data_channel_i,
                                                                hls::stream<block_t> stream_data_ch_0[LEVELS],
                                                                hls::stream<data_type> stream_data_ch_1[LEVELS],
                                                                int *enable_read){

#pragma HLS inline
ch_serialize_and_filter:
  for (int i = 0; i < LEVELS; i++) {
    #pragma HLS UNROLL
    serialize_and_filter(I_ITER, num_pixels, channel_blocks, channel_size, offset_read_data_channel_i[i], stream_data_ch_0[i], stream_data_ch_1[i], enable_read[i]);
  }
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
static void join(int H, int W, int I_ITER, int num_extra_rows, hls::stream<data_type> in[CPI], hls::stream<pixel_in_t> &out) {

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
      DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX)
      #pragma HLS PIPELINE II=1
      pixel_in_t data;
      DO_PRAGMA(HLS ARRAY_PARTITION variable=data complete dim=0)
      for(int i=0; i<CPI; i++){
        DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
        #pragma HLS UNROLL
        data.pixel[i] = in[i].read();
        #ifdef DEBUG_JOIN
        printf("data.pixel[%d] = %6.2f  ", i, float(data.pixel[i]));
        #endif
      }
      #ifdef DEBUG_JOIN
      printf("\n");
      #endif
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
static void split(int H, int W, hls::stream<pixel_out_t> &in, hls::stream<block_t> out[CPO]) {

  #ifdef DEBUG_SPLIT
  printf("SPLIT: starts (num pixels %d)\n", H * W);
  #endif

  int num_pixels = H * W;                                       // pixels to receive per channel
  int b = 0;
  block_t b_[CPO];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=b_ complete dim=0)
  pixel_out_t data;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=data complete dim=0)

  split_loop:
  for (int r=0; r<num_pixels; r++) {
    DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX)
    #pragma HLS PIPELINE II=1
    data = in.read();
    #ifdef DEBUG_SPLIT
    for(int i=0; i<CPO; i++){
      printf("data.pixel[%d] = %6.2f  ", i, float(data.pixel[i]));
    }
    printf("\n");
    #endif
    for(int i=0; i<CPO; i++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL
      b_[i].pixel[b] = data.pixel[i];
    }
    b = b + 1;
    if (b == BLOCK_SIZE  || (r==num_pixels-1)) {
      for(int i=0; i<CPO; i++){
        DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
        #pragma HLS UNROLL
        out[i] << b_[i];
      }
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
static void write_data_channels(int num_pixels, data_type *ptr, int *offset_i, hls::stream<block_t> in[CPO], int *enable_write) {

  #ifdef DEBUG_WRITE_DATA
  printf("WRITE_DATA: starts\n");
  #endif

  int num_blocks = num_pixels / BLOCK_SIZE;
  int res_blocks = num_pixels % BLOCK_SIZE;

  #ifdef DEBUG_WRITE_DATA
  printf("WRITE_DATA: num_blocks %d \n", num_blocks);
  printf("WRITE_DATA: res_blocks %d \n", res_blocks);
  printf("WRITE_DATA: pixels = %d\n", num_pixels);
  printf("WRITE_DATA: Offsets = ");
  for(int i=0; i<CPO; i++){
  printf("%d ", offset_i[i]);
  }
  printf("\n");
  #endif

  int offset;
  int channel = 0;
  int pos = 0;
  int slot = 0;
  int enable;
  block_t bx;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=bx complete)


  for (int p = 0; p < num_blocks * CPO; p++) {
    #pragma HLS pipeline II=1
    bx = in[channel].read();
    enable = enable_write[channel];
    offset = offset_i[channel];
    if(enable){
      for(int x = 0; x<BLOCK_SIZE; x++){
        #pragma HLS UNROLL
        #ifdef DEBUG_WRITE_DATA
        printf("Channel %d -- bx.pixel[ %d + %d] = %6.2f \n", channel, offset, x, float(bx.pixel[x]));
        #endif
        ptr[offset+x] = bx.pixel[x];
      }
    offset_i[channel] = offset_i[channel] + BLOCK_SIZE;
    }
    channel = (channel + 1) % CPO;
  }

  if(res_blocks != 0){
    for(int o =0 ; o<CPO; o++){
      bx = in[channel].read();
      enable = enable_write[channel];
      offset = offset_i[channel];
      if(enable){
        for(int x = 0; x<res_blocks; x++){
          #pragma HLS UNROLL
          #ifdef DEBUG_WRITE_DATA
            printf("Channel %d -- bx.pixel[ %d + %d] = %6.2f \n", channel, offset, x, float(bx.pixel[x]));
          #endif
          ptr[offset+x] = bx.pixel[x];
        }
        offset_i[channel] = offset_i[channel] + BLOCK_SIZE;
      }
      channel = (channel + 1) % CPO;
    }
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

  #ifdef DEBUG_RELU
  printf("relu: start\n");
  #endif

  pixel_out_t data;
  int data_size = W * H;
  for (int i=0; i < data_size; i++) {
    DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX)
    #pragma HLS PIPELINE II=1
    data  = in.read();
    for(int cpo = 0; cpo<CPO; cpo++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL
      if(enable_relu & (data.pixel[cpo] < 0)) data.pixel[cpo] = data_type(0.f);
    }
    out << data;
  }

  #ifdef DEBUG_RELU
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
  for (int cpi=0; cpi<CPI; cpi++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
    #pragma HLS UNROLL
    zero.pixel[cpi] = 0.f;
  }
  padding_iter_loop:
  for(int iter = 0; iter < I_ITER; iter++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=512/CPI)
    #pragma HLS PIPELINE II=1
    padding_h_loop:
    for(int h = 0; h < H + 2; h++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=HMAX+2)
      padding_w_loop:
      for(int w = 0; w < W + 2; w++){
        DO_PRAGMA(HLS loop_tripcount  min=1 max=WMAX+2)
        #pragma HLS PIPELINE II=1
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

  #ifdef DEBUG_CVT
  printf("cvt: start\n");
  #endif

  cvt_i_iter_loop:
  for(int i_iter = 0; i_iter < I_ITER; i_iter++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=512/CPI)

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
      DO_PRAGMA(HLS loop_tripcount  min=1 max=HMAX+2)
      cvt_loop_2:
      for (int pin_col=0; pin_col < W+2; pin_col++) {
        DO_PRAGMA(HLS loop_tripcount  min=1 max=WMAX+2)
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
          #ifdef DEBUG_CVT
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

  #ifdef DEBUG_CVT
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

  #ifdef DEBUG_MUL
  printf("mul: start\n");
  #endif

  kernel_t kernel;
  DO_PRAGMA(HLS ARRAY_PARTITION variable=kernel dim=0 complete)
  frame_t data_in;

  data_type sum[CPO];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=sum dim=0 block factor=CPO)

  pixel_out_t p_out;

  int load_kernel = 0;
  int num_iter = I_ITER * H * W;
  int iter_load_kernel = 0;

  mul_loop_1:
  for(int i = 0; i < num_iter; i++){
    DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX*512/CPI)
    #pragma HLS PIPELINE II=1
    load_kernel = (iter_load_kernel == 0);
    if (load_kernel){
      kernel = k_in.read();
      #ifdef DEBUG_MUL
      printf("MUL: kernel read\n");
      for(int i=0; i<CPI; i++){
        for(int o=0; o<CPO; o++){
          printf("kernel cpi=%d cpo=%d\n", i, o);
          for (int p=0; p<9; p++){
            printf(" %f ", float(kernel.pixel[o][i][p]));
            if((p+1)%3==0)printf("\n");
          }
          printf("\n");
        }
      }
      #endif
    }

    mul_loop_2:
    for(int i=0; i<CPO; i++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL
      sum[i] = 0;
    }

    data_in = in.read();

    loop_mul_cpi:
    for (int cpi=0; cpi<CPI; cpi++) {
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
      #pragma HLS UNROLL
      loop_mul_j:
      for (int j=0; j<KW*KH; j++) {
        DO_PRAGMA(HLS loop_tripcount  min=1 max=KW*KH)
        #pragma HLS UNROLL
        loop_mul_cpo:
        for (int cpo=0; cpo<CPO; cpo++) {
          DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
          #pragma HLS UNROLL
          sum[cpo] += data_in.pixel[j].pixel[cpi] * kernel.pixel[cpo][cpi][j];
        }
      }
    }

    for(int i=0; i<CPO; i++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL
      p_out.pixel[i] = sum[i];
    }
    #ifdef DEBUG_MUL
    for(int i = 0;i<CPO;i++) {
      printf("mult: p_out.pixel[%d] = %6.2f  ", i, float(p_out.pixel[i]));
    }
    printf("\n");
    #endif
    out << p_out;
    iter_load_kernel++;
    if (iter_load_kernel == W*H) iter_load_kernel = 0;
  }

  #ifdef DEBUG_MUL
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

  #ifdef DEBUG_ADD
  printf("add: start\n");
  #endif

  data_type bias[CPO];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=bias dim=0 complete)

  // number of iterations by CPI || CPO channels
  int num_iterations = W * H;

  // Buffer for all data and CPO channels
  data_type buff_o_channels[CPO][WHMAX];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=buff_o_channels dim=1 complete)


  // We receive bias in packs of CPO
  pixel_out_t p_out;
  p_out = b_in.read();
  add_load_bias_loop:
  for (int b=0; b<CPO; b++) {
    DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
    #pragma HLS PIPELINE II=1
    bias[b] = p_out.pixel[b];
  }

  #ifdef DEBUG_ADD
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
    DO_PRAGMA(HLS loop_tripcount  min=1 max=512/CPI)
    pixel_out_t data_out;
    #pragma HLS loop_flatten off
    add_load_data_it_loop:
    for(int it = 0; it<num_iterations; it++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=WHMAX)
      pixel_out_t data_in;
      data_in = in.read();
      pixel_out_t data;
      add_load_data_cpo_loop:
      for (int cpo=0; cpo<CPO; cpo++) {
        DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
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

  #ifdef DEBUG_ADD
  for (int cpo=0; cpo<CPO; cpo++) {
    printf("CH %d: ", cpo);
    for (int it=0; it<num_iterations; it++) {
      printf("%6.2f ", float(buff_o_channels[cpo][it]));
    }
    printf("\n");
  }
  #endif

  #ifdef DEBUG_ADD
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
  DO_PRAGMA(HLS stream variable=str_pad_cvt depth=CPO)
  DO_PRAGMA(HLS stream variable=str_cvt_mul depth=CPO)
  DO_PRAGMA(HLS stream variable=str_mul_add depth=CPO)


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

extern "C" {

void k_conv2D
                                (
		                             block_t *ptr_data,
                                 int H, int W, int rows, int I, int O, int I_ITER, int O_ITER, int enable_relu,
                                 kernel_in_t *ptr_kernel, pixel_out_t *ptr_bias,
                                 data_type *ptr_out,
                                 int global_offset, int enable_upper_padding, int enable_lower_padding
				) {

  #pragma HLS INTERFACE m_axi port=ptr_data offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi depth=10 port=ptr_kernel offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=ptr_bias offset=slave bundle=gmem2
  #pragma HLS INTERFACE m_axi port=ptr_out offset=slave bundle=gmem


  #ifdef DEBUG_VERBOSE
  printf("kernel starts...\n");
  #endif


  o_iter_loop:
  for (int o_iter = 0; o_iter<O_ITER; o_iter++) {
    DO_PRAGMA(HLS loop_tripcount  min=1 max=512/CPO)

    #pragma HLS dataflow
    // we compute the enable_write signals
    int shift = log2(CPO);
    int o_channel = (o_iter << shift);  // current output channel (first one in this iteration)
    int enable_write[CPO];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=enable_write dim=0 complete)
    init_enable_write_loop:
    for(int o = 0; o <CPO; o++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL
      enable_write[o] = (o_channel + o) < O;
    }

    // we compute the enable_read signals
    int enable_read[CPI];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=enable_read dim=0 complete)
    init_enable_read_loop:
    for(int i = 0; i <CPI; i++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
      #pragma HLS UNROLL
      enable_read[i] = (I >= i+1);
    }

    // input and output streams
    static hls::stream<pixel_in_t> out_read_data;
    DO_PRAGMA(HLS stream variable=out_read_data depth=CHUNK_SIZE)
    static hls::stream<kernel_t> out_read_kernel;
    static hls::stream<pixel_out_t> out_read_bias;
    static hls::stream<pixel_out_t> out_conv;
    DO_PRAGMA(HLS stream variable=out_conv depth=CPO)
    //ReLu stream
    static hls::stream<pixel_out_t> out_relu;
    DO_PRAGMA(HLS stream variable=out_relu depth=CPO)

    // array of stream declaration
    static hls::stream<block_t> stream_data_ch_0[CPI];
    DO_PRAGMA(HLS stream variable=stream_data_ch_0 depth=CHUNK_SIZE)
    static hls::stream<data_type> stream_data_ch_1[CPI];
    static hls::stream<block_t> out_write_channel[CPO];

    // channel offsets for reading
    int corrected_offset = (enable_upper_padding==0)? W : 0;
    int channel_offset = (W * H);
    int num_extra_rows = (enable_lower_padding == 0) + (enable_upper_padding == 0);
    int offset_read_data_channel = global_offset - corrected_offset;

    int channel_size = H * W;
    int read_pixels = W * (rows + num_extra_rows);
    int write_pixels = rows * W;
    int channel_blocks = (read_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int res_blocks = channel_size % BLOCK_SIZE;
    channel_blocks = (res_blocks != 0) ? channel_blocks + 1 : channel_blocks;
    printf("channel_size %d, read_pixels %d, write_pixels %d, channel_blocks = %d\n", channel_size, read_pixels, write_pixels, channel_blocks);

    // channel offsets for reading
    int offset_read_data_channel_i[CPI];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=offset_read_data_channel_i dim=0 complete)
    init_offset_read_data_channel_i_loop:
    for(int i=0; i<CPI; i++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPI)
      #pragma HLS UNROLL
      offset_read_data_channel_i[i] = (offset_read_data_channel + i * channel_offset) % BLOCK_SIZE;
    }

    // channel offsets for writing
    int offset_write_data_channel_i[CPO];
    DO_PRAGMA(HLS ARRAY_PARTITION variable=offset_write_data_channel_i dim=0 complete)
    init_offset_write_data_channel_i_loop:
    for(int i=0; i<CPO; i++){
      DO_PRAGMA(HLS loop_tripcount  min=1 max=CPO)
      #pragma HLS UNROLL
      offset_write_data_channel_i[i] = global_offset + (channel_offset * i) + ((o_iter << shift) * channel_offset);
    }

    // int offset_bias = o_iter << shift; //offset_bias for data_type pointer
    int offset_bias = o_iter; // offset_bias for pixel_out_t pointer
    // int offset_kernel = o_iter * (I < CPI ? CPI : I) * 9 * CPO; //offset_kernel for data_type pointer
    int offset_kernel = o_iter * (I < CPI ? CPI : I) * CPO; //offset_kernel for kernel_in_t pointer


    read_data_channels(H, W, rows, I_ITER, ptr_data, offset_read_data_channel, num_extra_rows, channel_blocks, stream_data_ch_0, enable_read);
    ch_serialize_and_filter<CPI>(I_ITER, read_pixels, channel_blocks, channel_size, offset_read_data_channel_i, stream_data_ch_0, stream_data_ch_1, enable_read);
    join(rows, W, I_ITER, num_extra_rows, stream_data_ch_1,  out_read_data);
    read_bias(offset_bias, ptr_bias, out_read_bias);
    read_kernel(I_ITER, offset_kernel, ptr_kernel, out_read_kernel);
    conv(rows, W, I_ITER, enable_upper_padding, enable_lower_padding, out_read_data, out_read_kernel, out_read_bias, out_conv);
    relu(enable_relu, rows, W, out_conv, out_relu);
    split(rows, W, out_relu, out_write_channel);
    write_data_channels(write_pixels, ptr_out, offset_write_data_channel_i, out_write_channel, enable_write);
  }

  #ifdef DEBUG_VERBOSE
  printf("kernel finishes\n");
  #endif
}

} // end extern "C"
