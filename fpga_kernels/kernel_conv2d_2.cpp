#include <math.h>
#include <stdio.h>
#include <ap_int.h>
#include <hls_stream.h>

//#define DEBUG_VERBOSE

extern "C" {

// Fixed parameters (optimized at compilation/synthesis time)
#define KW       3 // kernel width
#define KH       3 // kernel height
#define I        16 // number of input channels 
#define O        16 // number of output channels 
#define CPI      16 // channels per input port
#define CPO      16 // channels per output port
#define W        256 // input width
#define H        256 // input height

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
// read_input: Reads an input of format (H, W, I) and writes it into one output stream
// First, the function reads and writes the kernels and bias to the stream, then it 
// reads and writes the input data into the stream.
//
// Arguments:
//   ptr  : Pointer to input data
//   k_ptr: pointer to kernels
//   b_ptr: pointer to bias
//   out  : output stream
//
static void read_input(pixel_in_t *ptr, float *k_ptr, float *b_ptr, hls::stream<pixel_in_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("read_input: start\n");
#endif

  pixel_in_t p;
  #pragma HLS ARRAY_PARTITION variable=p dim=0

#ifdef LOAD_MODEL

#ifdef READ_MODEL
  // first we send the kernels data through the streams
  int kernel_size = I * O * KW * KH;
  read_loop_kernel_load:
    for (int s=0; s<kernel_size; s++) {
      #pragma HLS PIPELINE II=1
      float v = k_ptr[s];
      p.pixel[0] = v;
      out << p;
    }
#else
  p.pixel[0] = 1.f;
  for (int s=0; s<I*O*KW*KH; s++) {
    #pragma HLS PIPELINE II=1
    out << p;
  }
#endif

#ifdef DEBUG_VERBOSE
  printf("read_input: kernels loaded\n");
#endif

#ifdef READ_MODEL
  // now we send the bias
  int bias_size = O;
  read_loop_bias_load:
    for (int b=0; b<bias_size; b++) {
      #pragma HLS PIPELINE II=1
      float v = b_ptr[b];
      p.pixel[0] = v;
      out << p;
    }
#else
  p.pixel[0] = 1.f;
  for (int b=0; b<O; b++) {
    #pragma HLS PIPELINE II=1
    out << p;
  }
#endif

#ifdef DEBUG_VERBOSE
  printf("read_input: bias loaded\n");
#endif

#endif

#ifdef READ_INPUT
  // now we read the input data, we read it as a 2D table
  float buf[CPI];
  read_loop_data_load_i:
  for (int r=0; r<H*W; r++) {
    #pragma HLS PIPELINE II=1
    out  << ptr[r];
  }
#else
  // we generate the input data
  for (int i=0; i<CPI; i++) p.pixel[i] = 1.f;
  read_loop_data_load_r:
  for (int r=0; r<H*W; r++) {
    #pragma HLS PIPELINE II=1
    out << p;
  }
#endif

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

#ifdef LOAD_MODEL
  // first we distribute the kernels through the output streams 
  int kernel_size_per_input_port = KW * KH;
  for (int cpi=0; cpi<CPI; cpi++) {
    for (int cpo=0 ; cpo < CPO; cpo++) {
      for (int s=0; s<kernel_size_per_input_port; s++) {
        pixel_in_t p;
        p = in.read();
	out << p;
      }
    }
  }

#ifdef DEBUG_VERBOSE
  printf("padding: kernels forwarded\n");
#endif

  // now we send all bias through all output ports
  int bias_size = O;
  for (int b=0; b<bias_size; b++) {
    pixel_in_t p;
    p = in.read();
    out << p;
  }

#ifdef DEBUG_VERBOSE
  printf("padding: bias forwarded\n");
#endif

#endif

  pixel_in_t pixel_zero;
  for (int cpi=0; cpi<CPI; cpi++) pixel_zero.pixel[cpi] = 0.f;

  // We inject padding
  for (int w=0; w<W+2; w++) out << pixel_zero;

  // Now we pad the incoming data
  loop_padding_h:
  for (int h=0; h < H; h++) {
    #pragma HLS_PIPELINE II=1
    loop_padding_w:
    for (int w=0; w < W+2; w++) {
      #pragma HLS_PIPELINE II=1
      pixel_in_t p;
      #pragma HLS ARRAY_PARTITION variable=p
      if ((w==0) | (w==W+1)) out << pixel_zero; else out << in.read();
    }
  }

  // We inject padding
  for (int w=0; w<W+2; w++) out << pixel_zero;

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
  // writes must be performed with pixel_in_t struct
  pixel_in_t p;
  #pragma HLS ARRAY_PARTITION variable=p dim=1

  for (int i=0; i<H*W; i++) {
    #pragma HLS PIPELINE II=1
    ptr[i] = in.read();
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
// At the begining the function forwards CPI * KW * KH data items and O data items, which represent the
// kernels and bias. After this forwarding, the function performs its normal convert operation
//
// Arguments:
//   in  : input stream
//   out : output stream
//   id  : function id (for debugging)
static void cvt(hls::stream<pixel_in_t> &in, hls::stream<frame_t> &out, int id) {

#ifdef DEBUG_VERBOSE
  printf("cvt_%d: start\n", id);
#endif

#ifdef LOAD_MODEL

  // first we forward the kernels
  int num_kernels = KW * KH * I * O;
  pixel_in_t p;
  frame_t frame_k;
  #pragma HLS ARRAY_PARTITION variable=frame_k dim=0
  int s = 0;
  int cpi = 0;
  loop_cvt_kernels_k:
  for (int k=0; k<num_kernels; k++) {
    #pragma HLS PIPELINE II=1
    p = in.read();
    frame_k.pixel[s].pixel[cpi] = p.pixel[0];
    s = s + 1;
    if (s == 9) {
      s = 0;
      cpi = cpi + 1;
      if (cpi == CPI) {
        cpi = 0;
	out << frame_k;
      }
    }
  }

#ifdef DEBUG_VERBOSE
  printf("cvt_%d: kernels forwarded\n", id);
#endif

  // now we forward the bias (each bias in one frame at pos 0)
  int num_bias = O;
  loop_cvt_bias_b:
  for (int b=0; b<num_bias; b++) {
    #pragma HLS PIPELINE II=1
    frame_t frame;
    pixel_in_t p;
    p = in.read();
    frame.pixel[0].pixel[0] = p.pixel[0];
    out << frame;
  }

#ifdef DEBUG_VERBOSE
  printf("cvt_%d: bias forwarded\n", id);
#endif

#endif

  // Now we process the input data and convert the data into frames
  // First we read the first two rows and produce data, then for each
  // input row we produce new frames

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
  for (int pin_row=0; pin_row < H+2; pin_row++) {
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

#ifdef DEBUG_VERBOSE
  printf("cvt_%d: end\n", id);
#endif
}

// ----------------------------------------------------------------------------------------
// mul: This function performs the multiplication of an input frame with the stored kernels
// and sends the produced pixels. Before normal operation it receives its kernels and 
// forwards input bias
//
// Arguments:
//   in: input stream with incoming frames
//   out: output stream
//   id: function id (for debugging only)
//
static void mul(hls::stream<frame_t> &in, hls::stream<pixel_out_t> &out, int id) { 

#ifdef DEBUG_VERBOSE
  printf("mul_%d: start\n", id);
#endif

  // first we read the kernels
  frame_t kernel[CPO];
  #pragma HLS ARRAY_PARTITION variable=kernel dim=0
  frame_t data_in;

#ifdef LOAD_MODEL

  loop_mul_kernels_load_cpo:
  for (int cpo=0; cpo<CPO; cpo++) {
    #pragma HLS PIPELINE II=1
    kernel[cpo] = in.read();
  }

#ifdef DEBUG_VERBOSE
  printf("mul_%d: kernels received\n", id);
  for (int cpi=0; cpi < CPI; cpi++) {
    for (int cpo=0; cpo < CPO; cpo++) {
      printf("  cpi=%d, cpo=%d:\n", cpi, cpo);
      printf("    %6.4f %6.4f %6.4f\n", kernel[cpo].pixel[0].pixel[cpi], kernel[cpo].pixel[1].pixel[cpi], kernel[cpo].pixel[2].pixel[cpi]);
      printf("    %6.4f %6.4f %6.4f\n", kernel[cpo].pixel[3].pixel[cpi], kernel[cpo].pixel[4].pixel[cpi], kernel[cpo].pixel[5].pixel[cpi]);
      printf("    %6.4f %6.4f %6.4f\n", kernel[cpo].pixel[6].pixel[cpi], kernel[cpo].pixel[7].pixel[cpi], kernel[cpo].pixel[8].pixel[cpi]);
    }
  }
#endif

  // now we forward the bias
  int num_bias = O;
  loop_mul_bias_forward_b:
  for (int b=0; b<num_bias; b++) {
    #pragma HLS PIPELINE II=1
    frame_t f;
    f = in.read();
    pixel_out_t p_out;
    p_out.pixel[0] = f.pixel[0].pixel[0];
    out << p_out;
  }

#ifdef DEBUG_VERBOSE
  printf("mul_%d: bias forwarded\n", id);
#endif

#endif

  // now we read frames and produce the pixels
  //
  float sum[CPO];
  #pragma HLS ARRAY_PARTITION variable=sum dim=0 block factor=16

  int num_iterations = W * H;

  for (int cpo=0; cpo<CPO; cpo++) sum[cpo] = 0.f;
 
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
          sum[cpo] += data_in.pixel[j].pixel[cpi] * kernel[cpo].pixel[j].pixel[cpi];
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

#ifdef DEBUG_VERBOSE
  printf("mul_%d: end\n", id);
#endif
}

// -------------------------------------------------------------------------------
// add: This function performs the addition of all subpixels for the same channel.
// It adds also the corresponding bias. Before normal operation it receives all output
// bias
//
// Arguments:
//   in:  input streams
//   out: output stream
//
static void add(hls::stream<pixel_out_t> &in, hls::stream<pixel_out_t> &out) {

#ifdef DEBUG_VERBOSE
  printf("add: start\n");
#endif

  float bias[O];

#ifdef LOAD_MODEL

  // we read the bias
  int num_bias = O;
  for (int b=0; b<num_bias; b++) {
    pixel_out_t p_out;
    p_out = in.read();
    bias[b] = p_out.pixel[0];
  }

#ifdef DEBUG_VERBOSE
  printf("add: bias received\n");
#endif

#endif

  // Now we proceed with normal operation
  //
  // number of iterations
  int num_iterations = W * H;

  for (int it=0; it<num_iterations; it++) {
    pixel_out_t data_in;
    pixel_out_t data_out;
    data_in = in.read();
    for (int cpo=0; cpo<CPO; cpo++) data_out.pixel[cpo] = data_in.pixel[cpo] + bias[cpo];
    out << data_out;
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
static void conv(hls::stream<pixel_in_t> &in, hls::stream<pixel_out_t> &out) {

  // streams
  static hls::stream<pixel_in_t>  str_pad_cvt;  // padding->cvt
  static hls::stream<frame_t>     str_cvt_mul;  // cvt->mul
  static hls::stream<pixel_out_t> str_mul_add;  // mul->add

  // topology
  #pragma HLS dataflow
  padding(in, str_pad_cvt);          // padding
  cvt(str_pad_cvt, str_cvt_mul, 0);  // cvt
  mul(str_cvt_mul, str_mul_add, 0);  // mul
  add(str_mul_add, out);             // add
}

void k_conv2d_2(pixel_in_t *ptr_data, float *ptr_kernel, float *ptr_bias, pixel_out_t *ptr_out) {

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
  static hls::stream<pixel_out_t> out_conv;
  
  // stream sizes
  #pragma HLS STREAM variable = out_read depth = 32
  #pragma HLS STREAM variable = out_conv depth = 32
  #pragma HLS STREAM variable = out_relu depth = 32

  #pragma HLS dataflow
  read_input(ptr_data, ptr_kernel, ptr_bias, out_read);
  conv(out_read, out_conv);
  write_output(ptr_out, out_conv);
}

} // end extern "C"
