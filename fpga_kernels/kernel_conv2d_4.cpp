//KERNEL_CONV2D_4.cpp
//Modified by: Jorge García Martinez
//Date: 17/09/2020
//Description: Based on kenel_conv2d_3.cpp. The goal of this code is to perform convolutions with a large number of inputs
//and outputs.For this, we use iteratively a limited number of input and output channels in the kernel.
//In all functions are used two loops for output and input iterations. In add function is added a buffer which stores
//the data that It should be written into the memory.



#include <math.h>
#include <stdio.h>
#include <ap_int.h>

#include <hls_stream.h>

//#define DEBUG_VERBOSE

extern "C" {

// Fixed parameters (optimized at compilation/synthesis time)
#define KW       3 // kernel width
#define KH       3 // kernel height
#define I        4 //16 // number of input channels
#define O        4 //16 // number of output channels
#define CPI      4 //16 // channels per input port
#define CPO      4 //16 // channels per output port
#define W        256 // input width
#define H        256 // input height
#define I_ITER   I/CPI // itreciones por entrada
#define O_ITER   O/CPO // itreciones por entrada

#define LOAD_MODEL
#define READ_MODEL
#define READ_INPUT
#define WRITE_OUTPUT

// pixel_in
struct pixel_in_t {
  float pixel[CPI];
};

struct pixel_out_t {
  float pixel[CPO];
};

// frames struct
struct frame_t {
  pixel_in_t pixel[9];
};

// --------------------------------------------------------------------------------------
// read_input:
// The function reads and writes the kernels, bias and data in different stream.
// Data are sent to padding module, kenels to mul and bias to add modules.
// LOOP FLOW
// ko = 0
// b = 0
//   for o_iter 0 .. n
//        read bias[b..b+3]
//        b = b + 4
//        d = 0
//        ki = 0
//        for i_iter 0 .. n
//            read kernel[ki..ki+3][ko..ko+3]
//            ki = ki +4
//            read data[d..d+3]
//            d = d + 4
//
//        ko = ko + 4
//
//
// Arguments:
//   ptr  : Pointer to input data (in)
//   k_ptr: pointer to kernels (in)
//   b_ptr: pointer to bias (in)
//   out  : data output stream (out)
//   k_out: pointer to kernel (out)
//   b_out: pointer to bias (out)
//
static void read_input(pixel_in_t *ptr, float *k_ptr, float *b_ptr, hls::stream<frame_t> &k_out, hls::stream<pixel_out_t> &b_out, hls::stream<pixel_in_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("read_input: start\n");
#endif

  frame_t frame_k;
  #pragma HLS ARRAY_PARTITION variable=frame_k dim=0

  pixel_out_t bias;
  #pragma HLS ARRAY_PARTITION variable=bias dim=0

  pixel_in_t data;
  #pragma HLS ARRAY_PARTITION variable=data dim=0

  read_input_o_iter_loop:
  for (int o_iter = 0; o_iter < O_ITER; o_iter++){

    //Sending bias to add in pack of CPO bias
    read_loop_bias_load:
      for (int b=CPO*o_iter; b<CPO+CPO*o_iter; b++) {
        #pragma HLS PIPELINE II=1
        float v = b_ptr[b];
        bias.pixel[0] = v;
        b_out << bias;
      }
    read_input_i_iter_loop:
    for (int i_iter = 0; i_iter < I_ITER; i_iter++){
      // printf("o_iter = %d -- i_iter = %d \n ", o_iter, i_iter);
      //Sending kernels to mul in pack of CPI*CPO kernels
      int kernel_size_cpo = CPO*KH*KW; //kernels size each i_iter
      int i_offset = I_ITER * CPI * CPO * KH * KW; //addr_k offset for each i_iter
      int cpo = 0; //index for kernel size
      int kx = 0; //index for channels
      read_loop_kernel_load_ext:
      for(int i = 0; i < CPI; i++){
        // printf("i = %d -- kernel_size_cpo = %d \n", i, kernel_size_cpo);
        read_loop_kernel_load_int:
        for (int j = 0; j < kernel_size_cpo; j++) {
          int addr_k = j + i*kernel_size_cpo*I_ITER + i_iter*i_offset + o_iter*kernel_size_cpo;
          float v = k_ptr[addr_k];
          frame_k.pixel[kx].pixel[cpo] = v;

          #ifdef DEBUG_VERBOSE
          printf("[%d]:", addr_k);
          printf("%6.4f ", v);
          #endif

          kx = kx + 1;
          if (kx == 9) {
            // printf("\n");
            kx = 0;
            cpo = cpo + 1;
            if (cpo == CPO) {
              cpo = 0;
              k_out << frame_k;
            }
          }
        }
      }

    //Sending data to padding  in pack of CPI channels
    read_loop_data_load_i:
      for (int r=0; r<H*W; r++) {
        #pragma HLS PIPELINE II=1
        int addr_d = r*I_ITER + i_iter;
        data = ptr[addr_d];

        #ifdef DEBUG_VERBOSE
        printf("data.pixel[0] = %6.2f  ", data.pixel[0]);
        printf("data.pixel[1] = %6.2f  ", data.pixel[1]);
        printf("data.pixel[2] = %6.2f  ", data.pixel[2]);
        printf("data.pixel[3] = %6.2f  \n", data.pixel[3]);
        #endif

        out  << data;
      }

   }
}


#ifdef DEBUG_VERBOSE
  printf("read_input: end\n");
#endif
}

// ---------------------------------------------------------------------------------------
// padding. Adds padding to the input and forwards it through the output
//
// Arguments:
//   in                : input stream
//   out               : vector of output streams
//
static void padding(hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("padding: start\n");
#endif

//we init zero only first time

pixel_in_t data;
#pragma HLS ARRAY_PARTITION variable=data complete

pixel_in_t zero;
#pragma HLS ARRAY_PARTITION variable=data complete

for (int cpi=0; cpi<CPI; cpi++) zero.pixel[cpi] = 0.f;

padding_o_iter_loop:
for (int o_iter = 0; o_iter < O_ITER; o_iter++){
  #pragma HLS loop_flatten off
  padding_i_iter_loop:
  for(int i_iter = 0; i_iter < I_ITER; i_iter++){

    for(int h = 0; h < H + 2; h++){
      #pragma HLS_PIPELINE II=1
      for(int w = 0; w < W + 2; w++){
        #pragma HLS_PIPELINE II=1

        if(h==0 || h == H+1 || w == 0 || w == W+1){
          for(int cpi = 0; cpi < CPI; cpi++){
            data = zero;
          }
        }
        else data = in.read();
        out << data;
      }
    }

  }
}


#ifdef DEBUG_VERBOSE
  printf("padding: end\n");
#endif
}

// ---------------------------------------------------------------------------------------------
// relu. Performs the relu operation on an input stream and produces an output stream
// Arguments:
//
//   in: input stream
//   out: output stream
//
static void relu(hls::stream<float> &in, hls::stream<float> &out) {

#ifdef DEBUG_VERBOSE
  printf("relu: start\n");
#endif

  int data_size = W * H * O;
  for (int i=0; i < data_size; i++) {
    #pragma HLS PIPELINE II=1
    float data = in.read();
    if (data < 0) data = 0.f;
    out << data;
  }

#ifdef DEBUG_VERBOSE
  printf("relu: end\n");
#endif
}

// --------------------------------------------------------------------------------
// write_output: Writes data comming from one stream into memory
// LOOP FLOW:
//  for o_iter 0 .. n
//      write data[do .. do+3]
//
//  d = d + 4
//
// Arguments:
//   ptr: memory address pointer
//   in: input stream
//
static void write_output(pixel_out_t *ptr, hls::stream<pixel_out_t> &in) {

#ifdef DEBUG_VERBOSE
  printf("write_output: start\n");
#endif

#ifdef WRITE_OUTPUT

  write_output_o_iter_loop:
  for (int o_iter = 0; o_iter<O_ITER; o_iter++){
    //writes must be performed with pixel_in_t struct

    write_output_data_size_loop:
    for (int i=0; i<H*W; i++) {
      int addr_d = i*O_ITER + o_iter;
      ptr[addr_d] = in.read();
      #ifdef DEBUG_VERBOSE
      printf("o_iter = %d para i = %d \n", o_iter, i);
      printf("ptr--p.pixel[0] = %6.2f \n", p.pixel[0]);
      printf("ptr--p.pixel[1] = %6.2f \n", p.pixel[1]);
      printf("ptr--p.pixel[2] = %6.2f \n", p.pixel[2]);
      printf("ptr--p.pixel[3] = %6.2f \n\n", p.pixel[3]);
      #endif
    }
  }

#else
  pixel_out_t sal;
  for (int i=0; i<H*W; i++) in >> sal;
#endif

#ifdef DEBUG_VERBOSE
  printf("write_output: end\n");
#endif
}



// ---------------------------------------------------------------------------------------------------
// cvt: reads an input stream with an image of format (W, H, CPI) and writes an output stream
// in a 2D format based on (KW, KH). (SW=1, SH=1) stride is assumed and (PW=1, PH=1) padding is assumed.
// The function outputs data in the format (CPI, KW, KH).
//
// Arguments:
//   in  : input stream
//   out : output stream
//   id  : function id (for debugging)
static void cvt(hls::stream<pixel_in_t> &in, hls::stream<frame_t> &out, int id) {

#ifdef DEBUG_VERBOSE
  printf("cvt_%d: start\n", id);
#endif

cvt_o_iter_loop:
for (int o_iter = 0; o_iter < O_ITER; o_iter++){
  cvt_i_iter_loop:
  for(int i_iter = 0; i_iter < I_ITER; i_iter++){

  // Now we process the input data and convert the data into frames

  // buffers (keep three rows)
  pixel_in_t buffer0[W+2];
  pixel_in_t buffer1[W+2];
  pixel_in_t buffer2[W+2];
  #pragma HLS ARRAY_PARTITION variable=buffer0 cyclic dim=1 factor=2
  #pragma HLS ARRAY_PARTITION variable=buffer1 cyclic dim=1 factor=2
  #pragma HLS ARRAY_PARTITION variable=buffer2 cyclic dim=1 factor=2

  // frame
  frame_t frame;
  #pragma HLS ARRAY_PARTITION variable=frame

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
      printf("cvt_%d: frame sent:\n", id);
      for (int cpi=0; cpi<CPI; cpi++) {
        printf("  cpi %d:\n", cpi);
        printf("    %6.4f %6.4f %6.4f\n", frame.pixel[0].pixel[cpi], frame.pixel[1].pixel[cpi], frame.pixel[2].pixel[cpi]);
        printf("    %6.4f %6.4f %6.4f\n", frame.pixel[3].pixel[cpi], frame.pixel[4].pixel[cpi], frame.pixel[5].pixel[cpi]);
        printf("    %6.4f %6.4f %6.4f\n", frame.pixel[6].pixel[cpi], frame.pixel[7].pixel[cpi], frame.pixel[8].pixel[cpi]);
      }
      #endif
     }
    }
  }

 }
}


#ifdef DEBUG_VERBOSE
  printf("cvt_%d: end\n", id);
#endif
}

// ----------------------------------------------------------------------------------------
// mul: This function performs the multiplication of an input frame with the stored kernels
// and sends the produced pixels. Before normal operation it receives its kernels
// Arguments:
//   in: input stream with incoming data frames
//   k_in: input stream with kernels
//   out: output stream
//   id: function id (for debugging only)
//
static void mul(hls::stream<frame_t> &in, hls::stream<frame_t> &k_in, hls::stream<pixel_out_t> &out, int id) {

#ifdef DEBUG_VERBOSE
  printf("mul_%d: start\n", id);
#endif

  // first we read the kernels
  frame_t kernel[CPI];
  #pragma HLS ARRAY_PARTITION variable=kernel dim=0
  frame_t data_in;

#ifdef LOAD_MODEL

  mul_o_iter_loop:
  for (int o_iter = 0; o_iter < O_ITER; o_iter++){
    mul_i_iter_loop:
    for(int i_iter = 0; i_iter < I_ITER; i_iter++){
      //we load the kernels into pack of frames
      loop_mul_kernels_load_cpo:
      for (int cpi=0; cpi<CPI; cpi++) {
        #pragma HLS PIPELINE II=1
        kernel[cpi] = k_in.read();
      }

#ifdef DEBUG_VERBOSE
  printf("mul_%d: kernels received\n", id);
  for (int cpi=0; cpi < CPI; cpi++) {
    for (int cpo=0; cpo < CPO; cpo++) {
      printf("  cpi=%d, cpo=%d:\n", cpi, cpo);
      printf("    %6.4f %6.4f %6.4f\n", kernel[cpi].pixel[0].pixel[cpo], kernel[cpi].pixel[1].pixel[cpo], kernel[cpi].pixel[2].pixel[cpo]);
      printf("    %6.4f %6.4f %6.4f\n", kernel[cpi].pixel[3].pixel[cpo], kernel[cpi].pixel[4].pixel[cpo], kernel[cpi].pixel[5].pixel[cpo]);
      printf("    %6.4f %6.4f %6.4f\n", kernel[cpi].pixel[6].pixel[cpo], kernel[cpi].pixel[7].pixel[cpo], kernel[cpi].pixel[8].pixel[cpo]);
    }
  }
#endif


    // now we read frames and produce the pixels
    float sum[CPO];
    #pragma HLS ARRAY_PARTITION variable=sum dim=0 block factor=4
    //factor = 16
    //the array_partition factor in this case is assumed to be CPO value
    int num_iterations = W * H;
    mul_sum_loop:
    for (int cpo=0; cpo<CPO; cpo++) sum[cpo] = 0.f;

    mul_num_iterations_loop:
    for (int i=0; i<num_iterations; i++) {
      data_in = in.read();

#ifdef DEBUG_VERBOSE
  printf("mul_%d: data received\n", id);
  for (int cpi=0; cpi<CPI; cpi++) {
    printf("  cpi=%d\n", cpi);
    printf("    %6.4f %6.4f %6.4f\n", data_in.pixel[0].pixel[cpi], data_in.pixel[1].pixel[cpi], data_in.pixel[2].pixel[cpi]);
    printf("    %6.4f %6.4f %6.4f\n", data_in.pixel[3].pixel[cpi], data_in.pixel[4].pixel[cpi], data_in.pixel[5].pixel[cpi]);
    printf("    %6.4f %6.4f %6.4f\n", data_in.pixel[6].pixel[cpi], data_in.pixel[7].pixel[cpi], data_in.pixel[8].pixel[cpi]);
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
          sum[cpo] += data_in.pixel[j].pixel[cpi] * kernel[cpi].pixel[j].pixel[cpo];
        }
      }
    }
    pixel_out_t p_out;
    for (int cpo=0; cpo<CPO; cpo++) {
      #pragma HLS unroll
      #ifdef DEBUG_VERBOSE
      printf("mul_%d: pixel produced\n", id);
      for (int cpo=0; cpo<CPO; cpo++) printf("  cpo=%d -> %6.4f\n", cpo, sum[cpo]);
      #endif
      p_out.pixel[cpo] = sum[cpo];
      sum[cpo] = 0.f;
     }
     out << p_out;
   }
    }
  }

#endif


#ifdef DEBUG_VERBOSE
  printf("mul_%d: end\n", id);
#endif
}

// -------------------------------------------------------------------------------
// add: This function performs the addition of all subpixels for the same channel.
// It adds also the corresponding bias.
// LOOP FLOW
//   for o_iter 0 .. n
//        receive bias[b..b+3]
//        init buff_o_channels with bias
//        for i_iter 0 .. n
//            receive data[do..d+3]
//            buff_o_channels = buff_o_channels + data
//
//        for num_iterations
//            for CPO
//              send data to write module
//
// Arguments:
//   in:  input streams data
//   b_in: input stream bias
//   out: output stream
//
static void add(hls::stream<pixel_out_t> &in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("add: start\n");
#endif

  float bias[CPO];

  //number of iterations by CPI || CPO channels
  int num_iterations = W * H;

  //Buffer for all data and CPO channels
  float buff_o_channels[CPO][num_iterations];
  #pragma HLS ARRAY_PARTITION variable=buff_o_channels dim=0 block factor=4

  //We read Bias in O_iter packs of CPO size
  add_o_iter_loop:
  for (int o_iter = 0; o_iter<O_ITER; o_iter++){

    //We receive bias in packs of CPO
    add_load_bias_loop:
    for (int b=0; b<CPO; b++) {
      #pragma HLS PIPELINE II=1
      pixel_out_t p_out;
      p_out = b_in.read();
      bias[b] = p_out.pixel[0];
    }

    #ifdef DEBUG_VERBOSE
    for (int b=0; b<CPO; b++) {
      printf("Bias[%d] = %6.4f \n", b, bias[b]);
    }
    #endif

    #ifdef DEBUG_VERBOSE
    printf("add: bias received\n");
    #endif

    //It is necessary to reset the buffer each o_iter
    add_init_buff_o_channels_loop:
    for(int cpo = 0; cpo<CPO; cpo++){
      for(int it = 0; it<num_iterations; it++){
        buff_o_channels[cpo][it] = bias[cpo];
      }
    }

      #ifdef DEBUG_VERBOSE
      printf("o_iter = %d \n", o_iter);
      for(int cpo = 0; cpo<CPO; cpo++){
        printf("Channel cpo = %d: ", cpo);
        for(int it = 0; it<num_iterations; it++){
          printf("%6.2f ", buff_o_channels[cpo][it]);
        }
        printf("\n");
      }
      #endif

      //All input data have effect into output add
      add_i_iter_loop:
      for (int i_iter = 0; i_iter < I_ITER; i_iter++){
        #pragma HLS loop_flatten off
        add_load_data_it_loop:
        for(int it = 0; it<num_iterations; it++){

          pixel_out_t data_in;
          data_in = in.read();
          add_load_data_cpo_loop:
          for (int cpo=0; cpo<CPO; cpo++) {
            buff_o_channels[cpo][it] = buff_o_channels[cpo][it] + data_in.pixel[cpo];
          }
        }
      }

      #ifdef DEBUG_VERBOSE
      printf("CH %d: ", o_iter*CPO);
      for (int it=0; it<num_iterations; it++) {
        printf("%6.2f ", buff_o_channels[0][it]);
      }
      printf("\n");
      printf("CH %d: ", o_iter*CPO +1);
      for (int it=0; it<num_iterations; it++) {
        printf("%6.2f ", buff_o_channels[1][it]);
      }
      printf("\n");
      printf("CH %d: ", o_iter*CPO +2);
      for (int it=0; it<num_iterations; it++) {
        printf("%6.2f ", buff_o_channels[2][it]);
      }
      printf("\n");
      printf("CH %d: ", o_iter*CPO +3);
      for (int it=0; it<num_iterations; it++) {
        printf("%6.2f ", buff_o_channels[3][it]);
      }
      printf("\n");
      #endif

      pixel_out_t d_out;

      //now we write the buff_o_channels to write_output function
      add_write_data_it_loop:
      for(int it = 0; it < num_iterations; it++){
        add_write_data_cpo_loop:
        for(int cpo = 0; cpo < CPO; cpo++){
          d_out.pixel[cpo] = buff_o_channels[cpo][it];
        }
        out << d_out;
        #ifdef DEBUG_VERBOSE
        printf("d_out.pixel[0] = %6.2f \n", d_out.pixel[0]);
        printf("d_out.pixel[1] = %6.2f \n", d_out.pixel[1]);
        printf("d_out.pixel[2] = %6.2f \n", d_out.pixel[2]);
        printf("d_out.pixel[3] = %6.2f \n\n", d_out.pixel[3]);
        #endif
      }

  }


#ifdef DEBUG_VERBOSE
  printf("add: end\n");
#endif

}

// conv: Convolutional kernel
//
// Arguments:
//   in: input stream
//   out: output stream
static void conv(hls::stream<pixel_in_t> &in, hls::stream<frame_t> &k_in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

  // streams
  static hls::stream<pixel_in_t>  str_pad_cvt;  // padding->cvt
  static hls::stream<frame_t>     str_cvt_mul;  // cvt->mul
  static hls::stream<pixel_out_t> str_mul_add;  // mul->add


  // topology
  #pragma HLS dataflow
  padding(in, str_pad_cvt);          // padding
  cvt(str_pad_cvt, str_cvt_mul, 0);  // cvt
  mul(str_cvt_mul, k_in, str_mul_add, 0);  // mul
  add(str_mul_add, b_in, out);             // add
}

void k_conv2d_4(pixel_in_t *ptr_data, float *ptr_kernel, float *ptr_bias, pixel_out_t *ptr_out) {

  //#pragma HLS INTERFACE s_axilite port=W bundle=control
  //#pragma HLS INTERFACE s_axilite port=H bundle=control
  #pragma HLS INTERFACE m_axi port=ptr_data offset=slave bundle=gmem   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_kernel offset=slave bundle=gmem max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_bias offset=slave bundle=gmem   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_out  offset=slave bundle=gmem   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  // ptr_data struct to be packed as a single element vector (to improve memory read)
  // the compiler will do full structure access (all elements of structure)
  #pragma HLS data_pack variable = ptr_data
  #pragma HLS data_pack variable = ptr_out

  // input and output streams
  static hls::stream<pixel_in_t> out_read;
  static hls::stream<frame_t> out_read_kernel;
  static hls::stream<pixel_out_t> out_read_bias;
  static hls::stream<pixel_out_t> out_conv;

  // stream sizes
  #pragma HLS STREAM variable = out_read depth = 32
  #pragma HLS STREAM variable = out_read_kernel depth = 32
  #pragma HLS STREAM variable = out_read_bias depth = 32
  #pragma HLS STREAM variable = out_conv depth = 32
  #pragma HLS STREAM variable = out_relu depth = 32

  #pragma HLS dataflow
  read_input(ptr_data, ptr_kernel, ptr_bias, out_read_kernel, out_read_bias, out_read);
  conv(out_read, out_read_kernel, out_read_bias, out_conv);
  write_output(ptr_out, out_conv);
}

} // end extern "C"
