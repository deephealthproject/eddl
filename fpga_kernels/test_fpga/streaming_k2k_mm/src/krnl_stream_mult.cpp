/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

// #define DEBUG_VERBOSE

#define DWIDTH 32
typedef ap_axiu<DWIDTH, 0, 0, 0> pkt;

typedef union{
                float f;
                uint32_t i;
              } t_rip;

extern "C" {
void krnl_stream_mult(float *in1,              // Read-Only Vector 1
                      hls::stream<pkt> &out, // Internal Stream
                      int H,
                      int W,
                      float value,
                      int enable_mult
                      ) {
#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem0
#pragma HLS INTERFACE axis port = out
#pragma HLS INTERFACE s_axilite port = in1 bundle = control
#pragma HLS INTERFACE s_axilite port = H bundle = control
#pragma HLS INTERFACE s_axilite port = W bundle = control
#pragma HLS INTERFACE s_axilite port = value bundle = control
#pragma HLS INTERFACE s_axilite port = enable_mult bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifdef DEBUG_VERBOSE
printf("stream mult: start\n");
#endif

int size = H*W;
vmult:
  for (int i = 0; i < size; i++) {
    #pragma HLS PIPELINE II = 1
    float res = in1[i];
    #ifdef DEBUG_VERBOSE
    printf("%6.2f ", res);
    #endif
    if(enable_mult){ res = res * value;}
    t_rip temp;
    temp.f = res;
    pkt v;
    v.data = temp.i;
    out.write(v);
  }

#ifdef DEBUG_VERBOSE
printf("stream mult: end\n");
#endif

}
} // extern "C"
