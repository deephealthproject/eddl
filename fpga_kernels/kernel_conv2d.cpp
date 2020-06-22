#include <math.h>
#include <stdio.h>

#include "ap_int.h"

extern "C" {




// k_conv2d kernel with subchannels of a given size and with local memory

#define SCH_ROWS 16
#define SCH_COLS 16
#define LOG_SCH_COLS 4

void k_conv2d(int batch_size,
              float *I, int Irows, int Icols, int Ichannels,
              float *K, int Krows, int Kcols,
              float *B, int use_bias,
              float *O, int Orows, int Ocols, int Ochannels,
              int padding_rows, int padding_cols,
              int stride_rows, int stride_cols) {

  #pragma HLS INTERFACE m_axi port=I    offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=K    offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B    offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=O    offset=slave bundle=gmem

  #pragma HLS INTERFACE s_axilite port=batch_size bundle=control
  #pragma HLS INTERFACE s_axilite port=Irows bundle=control
  #pragma HLS INTERFACE s_axilite port=Icols bundle=control
  #pragma HLS INTERFACE s_axilite port=Ichannels bundle=control 
  #pragma HLS INTERFACE s_axilite port=Krows bundle=control
  #pragma HLS INTERFACE s_axilite port=Kcols bundle=control
  #pragma HLS INTERFACE s_axilite port=use_bias bundle=control
  #pragma HLS INTERFACE s_axilite port=Orows bundle=control     
  #pragma HLS INTERFACE s_axilite port=Ocols bundle=control
  #pragma HLS INTERFACE s_axilite port=Ochannels bundle=control
  #pragma HLS INTERFACE s_axilite port=padding_rows bundle=control
  #pragma HLS INTERFACE s_axilite port=padding_cols bundle=control
  #pragma HLS INTERFACE s_axilite port=stride_rows bundle=control
  #pragma HLS INTERFACE s_axilite port=stride_cols bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control    
  
  #pragma HLS INTERFACE s_axilite port=I         bundle=control
  #pragma HLS INTERFACE s_axilite port=K         bundle=control
  #pragma HLS INTERFACE s_axilite port=B         bundle=control
  #pragma HLS INTERFACE s_axilite port=O         bundle=control

  int I_BATCH_STRIDE = Irows * Icols * Ichannels;         // input batch size stride
  int I_CHANNEL_STRIDE = Irows * Icols;                   // input channel stride
  int I_SCH_COL_STRIDE = SCH_COLS;                        // input subchannel col stride
  int I_SCH_ROW_STRIDE = SCH_ROWS * Icols;                // input subchannel row stride
  int I_COL_STRIDE  = 1;                                  // input column stride
  int I_ROW_STRIDE  = Icols;                              // input row stride
  int O_BATCH_STRIDE = Orows * Ocols * Ochannels;         // output batch size stride
  int O_CHANNEL_STRIDE = Orows * Ocols;                   // output channel stride
  int O_SCH_COL_STRIDE = SCH_COLS;                        // output channel stride
  int O_SCH_ROW_STRIDE = SCH_ROWS * Icols;                // output channel stride
  int O_COL_STRIDE  = 1;                                  // output column stride
  int O_ROW_STRIDE  = Ocols;                              // output row stride
  int KERNEL_ICHANNEL_STRIDE= Kcols * Krows;              // kernel input channel stride
  int KERNEL_OCHANNEL_STRIDE= Kcols * Krows * Ichannels;  // kernel output channel stride
  int KERNEL_ROW_STRIDE  = Kcols;                         // kernel row stride
  int KERNEL_COL_STRIDE  = 1;                             // kernel col stride 

  int Osubchannels_rows = Orows / SCH_ROWS;
  int Osubchannels_cols = Ocols / SCH_COLS;

//  printf("Irows %d Icols %d\n", Irows, Icols);

  // we sweep all batches
  for (int b=0; b < batch_size; b++) {

//	  printf("batch %d\n", b);

    // we sweep all output channels
    for (int och=0; och < Ochannels; och++) {

//	    printf("  och %d\n", och);

      // we sweep all subchannels in rows
      for (int osch_r=0; osch_r < Osubchannels_rows; osch_r++) {
	int isch_r = osch_r;


//	printf("    osch_r %d\n", osch_r);

        // we sweep all subchannels in cols
	for (int osch_c=0; osch_c < Osubchannels_cols; osch_c++) {
	  int isch_c = osch_c;


//	  printf("    osch_c %d\n", osch_c);

          // We initialize the output grid
	  float output[SCH_ROWS][SCH_COLS];
	  for (int r=0; r < SCH_ROWS; r++) {
	    for (int c=0; c < SCH_COLS; c++) {
	      output[r][c] = 0.f;
	    }
	  }

//	  printf("output initialized\n");

	  // we sweep all input channels
	  for (int ich=0; ich < Ichannels; ich++) {
	    int addr_kernel = (ich * KERNEL_ICHANNEL_STRIDE) + (och * KERNEL_OCHANNEL_STRIDE);


//	    printf("      ich %d\n", ich);

            // we read the kernel for the input channel and the output channel
	    float kernel[3][3];
	    for (int r=0; r<3; r++) {
	      for (int c=0; c<3; c++) {
	        kernel[r][c] = K[addr_kernel];
		addr_kernel = addr_kernel + 1;
              }
	    }

//	    printf("kernel read\n");

	    // now we read all the input subchannel and perform all multiplications
	    float A[SCH_ROWS+2][SCH_COLS+2];


	    int first_row;
	    if (isch_r == 0) {
	      for (int c=0; c< SCH_COLS+1; c++) A[0][c] = 0.f;
	      first_row = 1;
	    } else {
	      first_row = 0;
	    }

	    int last_row;
	    if (isch_r == Osubchannels_rows) {
	      for (int c=0; c<SCH_COLS+1; c++) A[SCH_ROWS-1][c] = 0.f;
	      last_row = SCH_ROWS-2;
	    } else {
	      last_row = SCH_ROWS-1;
	    }

	    int first_col;
	    if (isch_c == 0) {
	      for (int r=0; r<SCH_ROWS+1; r++) A[r][0] = 0.f;
	      first_col = 1;
	    } else {
	      first_col = 0;
	    }

	    int last_col;
	    if (isch_c == Osubchannels_cols) {
	      for (int r=0; r<SCH_ROWS+1; r++) A[r][SCH_COLS-1] = 0.f;
	      last_col = SCH_COLS-2;
	    } else {
	      last_col = SCH_COLS-1;
	    }

	    int addr_i_init = ich * I_CHANNEL_STRIDE + isch_r * I_SCH_ROW_STRIDE + isch_c * I_SCH_COL_STRIDE + b * I_BATCH_STRIDE;
	    int addr_i = addr_i_init;
	    int row = 0;
	    for (int r=first_row; r <= last_row; r++) {
	      int addr_i = addr_i_init + row * I_ROW_STRIDE;
	      for (int c=first_col; c <= last_col; c++) {
		  A[r][c] = I[addr_i];
		  addr_i = addr_i + 1;
              }
	      row++;
	    }

//	    printf("activations read\n");

	    // now we perform the computation
	    for (int r=0; r < SCH_ROWS; r++) {
	      #pragma HLS PIPELINE II=1
              #pragma HLS UNROLL FACTOR=2
	      for (int c=0; c < SCH_COLS; c++) {
		for (int kr=0; kr < 3; kr++) {
	          for (int kc=0; kc < 3; kc++) {
		    int row = r + 1 - kr;
		    int col = c + 1 - kc;
		    int multiply = (row >=0) && (col>=0) && (row!=16) && (col!=16);
		    if (multiply) output[row][col] += A[r][c] * kernel[kr][kc];
	          }
		}
	      }
	    }
          }

	  int addr_o_init = och * O_CHANNEL_STRIDE + osch_r * O_SCH_ROW_STRIDE + osch_c * O_SCH_COL_STRIDE + b * O_BATCH_STRIDE;
          // we write the output subchannel
	  for (int r=0; r < SCH_ROWS; r++) {
	    int addr_o = addr_o_init + r * O_ROW_STRIDE;
            for (int c=0; c < SCH_ROWS; c++) {
	      O[addr_o] = output[r][c];
	      addr_o = addr_o + 1;
	    }
	  }
        }
      }
    }
  }
}

} // end extern "C"
