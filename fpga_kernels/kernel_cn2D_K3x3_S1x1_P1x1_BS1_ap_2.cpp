//KERNEL_CONV2D_4.cpp
//Modified by: Jorge García Martinez
//Date: 17/09/2020
//Description: Based on kenel_conv2d_3.cpp. The goal of this code is to perform convolutions with a large number of inputs
//and outputs.For this, we use iteratively a limited number of input and output channels in the kernel.
//In all functions are used two loops for output and input iterations. In add function is added a buffer which stores
//the data that It should be written into the memory.



#include <math.h>
#include <stdio.h>
#include <ap_fixed.h>

#include <hls_stream.h>

// #define DEBUG_VERBOSE

extern "C" {

//#define data_type ap_fixed<8,4,AP_TRN,AP_WRAP>
#define data_type float

// To allow using defines inside Xilinx pragmas
#define PRAGMA_SUB(x) _Pragma (#x)
#define DO_PRAGMA(x) PRAGMA_SUB(x)

// Fixed parameters (optimized at compilation/synthesis time)
#define KW       3  // kernel width
#define KH       3  // kernel height
#define CPI      4  // channels per input port
#define CPO      4  // channels per output port

#define WMAX 256
#define WHMAX 256*256

#define LOAD_MODEL
#define READ_MODEL
#define READ_INPUT
#define WRITE_OUTPUT

// pixel_in
struct pixel_in_t {
  data_type pixel[CPI];
};

struct pixel_out_t {
  data_type pixel[CPO];
};

// frames struct
struct frame_t {
  pixel_in_t pixel[9];
};

// ---------------------------------------------------------------------------------------
// read_bias. Reading bias from memory and sending to add module.
//
// Arguments:
//   b_ptr                : pointer to bias
//   b_out               :  output streams
//
static void read_bias(int offset_bias, data_type *b_ptr, hls::stream<pixel_out_t> &b_out){

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
// read_kernel. Adds padding to the input and forwards it through the output
//
// Arguments:
//   k_ptr                : pointer to kernels
//   k_out               :  output stream
//
static void read_kernel(int I_ITER, int offset_kernel, data_type *k_ptr, hls::stream<frame_t> &k_out){

#ifdef DEBUG_VERBOSE
  printf("read_kernel: start\n");
#endif

  // we read all the kernels and send it through the stream
  frame_t frame_k;
  #pragma HLS ARRAY_PARTITION variable=frame_k dim=0
  int cpo = 0;
  int p = 0;

  int size = KW * KH * CPO * I_ITER * CPI;
  read_kernel_loop:
  for (int i=0; i<size; i++) {
    frame_k.pixel[p].pixel[cpo] = k_ptr[i+ offset_kernel];
    p = p + 1;
    if (p == 9) {
      p = 0;
      cpo = cpo+1;
      if (cpo == CPO) {
        cpo = 0;
	      k_out << frame_k;
#ifdef DEBUG_VERBOSE
	printf("kernel read:\n");
	for (int c=0; c<CPO; c++) {
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

// --------------------------------------------------------------------------------------
// read_data: Reading data from memory and sending to conv module
// Arguments:
//   ptr  : Pointer to input data (in)
//   out  : data output stream (out)
//
static void read_data(int H, int W, int I_ITER, pixel_in_t *ptr, hls::stream<pixel_in_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("read_data: start\n");
#endif


    read_loop_data_load_i:
      for (int r=0; r<H*W*I_ITER; r++) {
      	#pragma HLS PIPELINE II=1
      	pixel_in_t data;
      	data = ptr[r];
        #ifdef DEBUG_VERBOSE
        	printf("read data:\n");
                for(int cpi = 0;cpi<CPI;cpi++) printf(" %f ", float(data.pixel[cpi]));
        	printf("\n");
        #endif
        out  << data;
      }


#ifdef DEBUG_VERBOSE
  printf("read_data: end\n");
#endif
}

// ---------------------------------------------------------------------------------------
// padding. Adds padding to the input and forwards it through the output
//
// Arguments:
//   in                : input stream
//   out               : vector of output streams
//
static void padding(int H, int W, int I_ITER, hls::stream<pixel_in_t> &in, hls::stream<pixel_in_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("padding: start\n");
#endif

//we init zero only first time

pixel_in_t data;
DO_PRAGMA(HLS ARRAY_PARTITION variable=data complete)

pixel_in_t zero;
DO_PRAGMA(HLS ARRAY_PARTITION variable=zero complete)

for (int cpi=0; cpi<CPI; cpi++) zero.pixel[cpi] = 0.f;

  padding_iter_loop:
  for(int iter = 0; iter < I_ITER; iter++){

    for(int h = 0; h < H + 2; h++){
      #pragma HLS_PIPELINE II=1
      for(int w = 0; w < W + 2; w++){
        #pragma HLS_PIPELINE II=1
        if (h==0 || h == H+1 || w == 0 || w == W+1) {
          data = zero;
        }
        else {
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

// ---------------------------------------------------------------------------------------------
// relu. Performs the relu operation on an input stream and produces an output stream
// Arguments:
//
//   in: input stream
//   out: output stream
//
static void relu(int H, int W, int O, hls::stream<data_type> &in, hls::stream<data_type> &out) {

#ifdef DEBUG_VERBOSE
  printf("relu: start\n");
#endif

  int data_size = W * H * O;
  for (int i=0; i < data_size; i++) {
    #pragma HLS PIPELINE II=1
    data_type data = in.read();
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
static void write_output(int H, int W, int offset_data_out, pixel_out_t *ptr, hls::stream<pixel_out_t> &in) {

#ifdef DEBUG_VERBOSE
  printf("write_output: start\n");
#endif


    write_output_data_size_loop:
    for (int i=0; i<H*W; i++) {
      pixel_out_t p = in.read();
      ptr[i + offset_data_out] = p;
      #ifdef DEBUG_VERBOSE
      printf("i = %d \n",  i);
      for (int cpo=0; cpo<CPO; cpo++) printf("ptr--p.pixel[%d] = %6.2f \n", cpo, float(p.pixel[cpo]));
      #endif
    }


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
static void cvt(int H, int W, int I_ITER, hls::stream<pixel_in_t> &in, hls::stream<frame_t> &out, int id) {

#ifdef DEBUG_VERBOSE
  printf("cvt_%d: start\n", id);
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
      printf("cvt_%d: frame sent:\n", id);
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
static void mul(int H, int W, int I_ITER, hls::stream<frame_t> &in, hls::stream<frame_t> &k_in, hls::stream<pixel_out_t> &out, int id) {

#ifdef DEBUG_VERBOSE
  printf("mul_%d: start\n", id);
#endif

  // first we read the kernels
  frame_t kernel[CPO];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=kernel dim=0)
  frame_t data_in;

#ifdef LOAD_MODEL

    mul_i_iter_loop:
    for(int i_iter = 0; i_iter < I_ITER; i_iter++){
      //we load the kernels into pack of frames
      loop_mul_kernels_load_cpo:
      for (int cpo=0; cpo<CPO; cpo++) {
        #pragma HLS PIPELINE II=1
        kernel[cpo] = k_in.read();
      }

#ifdef DEBUG_VERBOSE
  printf("mul_%d: kernels received\n", id);
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
      printf("mul_%d: pixel produced\n", id);
      for (int cpo=0; cpo<CPO; cpo++) printf("  cpo=%d -> %6.4f\n", cpo, float(sum[cpo]));
      #endif
      p_out.pixel[cpo] = sum[cpo];
      sum[cpo] = 0.f;
     }
     out << p_out;
    }
  } //i_iter

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
static void add(int H, int W, int I_ITER, hls::stream<pixel_out_t> &in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("add: start\n");
#endif

  data_type bias[CPO];

  //number of iterations by CPI || CPO channels
  int num_iterations = W * H;

  //Buffer for all data and CPO channels
  data_type buff_o_channels[CPO][WHMAX];
  DO_PRAGMA(HLS ARRAY_PARTITION variable=buff_o_channels dim=0 block factor=CPO)

    //We receive bias in packs of CPO
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
    #endif

    #ifdef DEBUG_VERBOSE
    printf("add: bias received\n");
    #endif


      #ifdef DEBUG_VERBOSE
      printf("o_iter = %d \n", o_iter);
      for(int cpo = 0; cpo<CPO; cpo++){
        printf("Channel cpo = %d: ", cpo);
        for(int it = 0; it<num_iterations; it++){
          printf("%6.2f ", float(buff_o_channels[cpo][it]));
        }
        printf("\n");
      }
      #endif

      //All input data have effect into output add
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
            }
            else{
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
// conv: Convolutional kernel
//
// Arguments:
//   in: input stream
//   out: output stream
static void conv(int H, int W, int I_ITER, hls::stream<pixel_in_t> &in, hls::stream<frame_t> &k_in, hls::stream<pixel_out_t> &b_in, hls::stream<pixel_out_t> &out) {

  // streams
  static hls::stream<pixel_in_t>  str_pad_cvt;  // padding->cvt
  static hls::stream<frame_t>     str_cvt_mul;  // cvt->mul
  static hls::stream<pixel_out_t> str_mul_add;  // mul->add


  // topology
  #pragma HLS dataflow
  padding(H, W, I_ITER, in, str_pad_cvt);          // padding
  cvt(H, W, I_ITER, str_pad_cvt, str_cvt_mul, 0);  // cvt
  mul(H, W, I_ITER, str_cvt_mul, k_in, str_mul_add, 0);  // mul
  add(H, W, I_ITER, str_mul_add, b_in, out);             // add
}


void k_cn2D_K3x3_S1x1_P1x1_BS1_ap_2(pixel_in_t *ptr_data, int H, int W, int I, data_type *ptr_kernel, data_type *ptr_bias, pixel_out_t *ptr_out, int O, int offset_bias, int offset_kernel, int offset_data_out) {

  #pragma HLS INTERFACE s_axilite port=W bundle=control
  #pragma HLS INTERFACE s_axilite port=H bundle=control
  #pragma HLS INTERFACE s_axilite port=I bundle=control
  #pragma HLS INTERFACE s_axilite port=O bundle=control
  #pragma HLS INTERFACE m_axi port=ptr_data offset=slave bundle=gmem  max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_kernel offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_bias offset=slave bundle=gmem2   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE m_axi port=ptr_out  offset=slave bundle=gmem   max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS INTERFACE s_axilite port=offset_bias bundle=control
  #pragma HLS INTERFACE s_axilite port=offset_kernel bundle=control
  #pragma HLS INTERFACE s_axilite port=offset_data_out bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  // ptr_data struct to be packed as a single element vector (to improve memory read)
  // the compiler will do full structure access (all elements of structure)
  #pragma HLS data_pack variable = ptr_data
  #pragma HLS data_pack variable = ptr_out

  int I_ITER = I/CPI;

  // input and output streams
  static hls::stream<pixel_in_t> out_read_data;
  static hls::stream<frame_t> out_read_kernel;
  static hls::stream<pixel_out_t> out_read_bias;
  static hls::stream<pixel_out_t> out_conv;

  // stream sizes
  #pragma HLS STREAM variable = out_read_data depth = 32
  #pragma HLS STREAM variable = out_read_kernel depth = 32
  #pragma HLS STREAM variable = out_read_bias depth = 32
  #pragma HLS STREAM variable = out_conv depth = 32
  // #pragma HLS STREAM variable = out_relu depth = 32

    #pragma HLS dataflow
    read_data(H, W, I_ITER, ptr_data, out_read_data);
    read_bias(offset_bias, ptr_bias, out_read_bias);
    read_kernel(I_ITER, offset_kernel, ptr_kernel, out_read_kernel);
    conv(H, W, I_ITER, out_read_data, out_read_kernel, out_read_bias, out_conv);
    write_output(H, W, offset_data_out, ptr_out, out_conv);

}

} // end extern "C"
