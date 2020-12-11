// --------------------------------------------------------------------------------------------------------------
// FPGA kernels for EDDL Library - European Distributed Deep Learning Library.
// Version: 0.6
// copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
// Date: December 2020
// Authors: GAP Research Group (UPV)
//     José Flich Cardo
//     Jorge García Martínez
//     Izan Catalán Gallarch
//     Carles Hernández Luz
//
// contact: jflich@disca.upv.es
// All rights reserved

// Includes
#include <ap_int.h>
#include <hls_stream.h>
#include <math.h>
#include <stdio.h>


#define max_memory_size 8 //max size of the input values

extern "C" {

static void read_input(float *in, hls::stream<float> &in_data_stream,  int size) {
  mem_rd:
  for (int i = 0; i < size; i++) {
      in_data_stream << in[i];
    }
}

static void read_values(float *values, hls::stream<float> &values_stream, int enable_read_from_ddr, int size){

  static float values_memory[max_memory_size]; // config values_memory vector as a RAM memory

  if(enable_read_from_ddr){
    read_values:
    for (int i = 0; i < size; i++) {
      printf("read values from DDR \n");
      #pragma HLS pipeline
      values_memory[i] = values[i];
      values_stream << values_memory[i];
    }
  }
  else{
    send_values:
    for (int i = 0; i < size; i++) {
      printf("read values from internal memory \n");
      values_stream << values_memory[i];
    }
  }
}

static void mult_passive(hls::stream<float> &in_data_stream,  hls::stream<float> &out_data_stream, hls::stream<float> &values_stream, int size) {
  mult_passive:
    for (int i = 0; i < size; i++) {
      #pragma HLS pipeline
      float a = in_data_stream.read();
      float b = values_stream.read();
      printf("a = %6.2f - value = %6.2f\n", a, b);
      out_data_stream << a*b;
    }
}


static void write_result(float *out, hls::stream<float> &out_data_stream, int size) {
  mem_wr:
    for (int i = 0; i < size; i++) {
      out[i] = out_data_stream.read();
    }
}

void k_mult_passive(float *in, float *out, float *values, int enable_read_from_ddr, int size) {

  #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=out  offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=values  offset=slave bundle=gmem1
  #pragma HLS INTERFACE s_axilite port=in bundle=control
  #pragma HLS INTERFACE s_axilite port=out bundle=control
  #pragma HLS INTERFACE s_axilite port=values bundle=control
  #pragma HLS INTERFACE s_axilite port=enable_read_from_ddr bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  printf("kenel mult passive\n");


  static hls::stream<float> in_data_stream("input_stream");
  static hls::stream<float> out_data_stream("output_stream");
  static hls::stream<float> values_stream("values_stream");
  #pragma HLS STREAM variable = in_data_stream depth = 32
  #pragma HLS STREAM variable = out_data_stream depth = 32
  #pragma HLS STREAM variable = values_stream depth = 32


  #pragma HLS dataflow
  read_input(in, in_data_stream, size);
  read_values(values, values_stream, enable_read_from_ddr, size);
  mult_passive(in_data_stream, out_data_stream, values_stream, size);
  write_result(out, out_data_stream, size);
 }
}//extern "C"
