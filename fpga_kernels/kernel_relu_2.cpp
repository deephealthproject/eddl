#include <math.h>
#include <stdio.h>
#include <ap_int.h>
#include <hls_stream.h>



#define DATA_SIZE 1024*128

// TRIPCOUNT identifier
const int c_size = DATA_SIZE;



// Read Data from Global Memory and write into Stream inStream
static void read_input(float *A, hls::stream<float> &inStream, long int size) {
// Auto-pipeline is going to apply pipeline to this loop
  mem_rd:
    for (int i = 0; i < size; i++) {
      #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
      // Blocking write command to inStream
      inStream << A[i];
    }
}

// Read Input data from inStream and write the result into outStream
static void compute_relu(hls::stream<float> &inStream, hls::stream<float> &outStream, long int size) {
// Auto-pipeline is going to apply pipeline to this loop
  relu:
    for (int i = 0; i < size; i++) {
      #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
      // Blocking read command from inStream and Blocking write command
      // to outStream
      float val = 0;
      val = inStream.read();
      if(val < 0){
        outStream << 0.0;
      }
      else{
        outStream << val;
      }
      // outStream << (inStream.read() + 1);
    }
}


// Read result from outStream and write the result to Global Memory
static void write_result(float *B,  hls::stream<float> &outStream, long int size) {
// Auto-pipeline is going to apply pipeline to this loop
mem_wr:
  for (int i = 0; i < size; i++) {
    #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
    // Blocking read command to inStream
    B[i] = outStream.read();
  }
}

extern "C" {

void k_relu(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  // Adding names for the streams. It allows the name to be used in reporting.
    // Vivado HLS
    // automatically checks to ensure all elements from an input stream are read
    // during sw emulation.
    static hls::stream<float> inStream("input_stream");
    static hls::stream<float> outStream("output_stream");
  #pragma HLS STREAM variable = inStream depth = 32
  #pragma HLS STREAM variable = outStream depth = 32
  //  HLS STREAM variable=<name> depth=<size> pragma is used to define the Stream
  //  depth. For this example, Depth 32 is defined. Which means that Stream can
  //  hold
  //  maximum 32 outstanding elements at a given time. If Stream is full, any
  //  further
  //  blocking write command from producer will go into wait state until consumer
  //  reads some elements from stream. Similarly if Stream is empty (no element in
  //  Stream)
  //  any blocking read command from consumer will go into wait state until
  //  producer
  //  writes elements to Stream. This blocking read and write allow consumer and
  //  producer to synchronize each other.

  #pragma HLS dataflow
    // dataflow pragma instruct compiler to run following three APIs in parallel
    read_input(A, inStream, size);
    compute_relu(inStream, outStream, size);
    write_result(B, outStream, size);


}
} // end extern "C"
