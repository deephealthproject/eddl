#include <math.h>
#include <stdio.h>
extern "C" {

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

//  #pragma HLS INTERFACE s_axilite port=I         bundle=control
//  #pragma HLS INTERFACE s_axilite port=K         bundle=control
//  #pragma HLS INTERFACE s_axilite port=B         bundle=control
//  #pragma HLS INTERFACE s_axilite port=O         bundle=control

  int IBS_STRIDE = Irows * Icols * Ichannels; // input batch size stride
  int ICH_STRIDE = Irows * Icols;             // input channel stride
  int IC_STRIDE  = 1;                         // input column stride
  int IR_STRIDE  = Icols;                     // input row stride
  int OBS_STRIDE = Orows * Ocols * Ochannels; // output batch size stride
  int OCH_STRIDE = Orows * Ocols;             // output channel stride
  int OC_STRIDE  = 1;                         // output column stride
  int OR_STRIDE  = Ocols;                     // output row stride
  int KICH_STRIDE= Kcols * Krows;             // kernel input channel stride
  int KOCH_STRIDE= Kcols * Krows * Ichannels; // kernel output channel stride
  int KC_STRIDE  = 1;                         // kernel column stride
  int KR_STRIDE  = Kcols;                     // kernel row stride 

  int i_addr;
  int k_addr;
  int o_addr;

  printf("batch size %d, Irows %d, Icols %d, Ichannels %d, Krows %d, Kcols %d, use_bias %d, Orows %d, Ocols %d, Ochannels %d, padding_rows %d, padding_cols %d stride_rows %d, stride_cols %d\n", batch_size, Irows, Icols, Ichannels, Krows, Kcols, use_bias, Orows, Ocols, Ochannels, padding_rows, padding_cols, stride_rows, stride_cols);

  // we sweep all batches
  for (int b=0; b<batch_size; b++) {

    int i_addr_1 = b * IBS_STRIDE;
    int o_addr_1 = b * OBS_STRIDE;

    // we sweep all output channels
    for (int och=0; och<Ochannels; och++) {

      int k_addr_1 = och * KOCH_STRIDE;
      int o_addr_2 = o_addr_1 + (och * OCH_STRIDE);

      // we seep all input channels
      for (int ich=0; ich<Ichannels; ich++) {

	int i_addr_2 = i_addr_1 + (ich * ICH_STRIDE);
	int k_addr_2 = k_addr_1 + (ich * KICH_STRIDE);

	int o_c = 0; // output column

        // we sweep all input cols
        for (int i_c=0; i_c<Icols; i_c=i_c+stride_cols) {

	  int o_r = 0;  // output row

	  int o_addr_3 = o_addr_2 + (o_c * OC_STRIDE);

          // we seep all input rows
          for (int i_r=0; i_r<Irows; i_r=i_r+stride_rows) {

            // we sweep all kernel cols
	    float out = 0.f;

	    for (int k_c=0; k_c<Kcols; k_c++) {

              int i_addr_3 = i_addr_2 + ((i_c + k_c) * IC_STRIDE);
	      int k_addr_3 = k_addr_2 + (k_c * KC_STRIDE);

	      // we seep all kernel rows
	      for (int k_r=0; k_r<Krows; k_r++) {

		//printf("och %d, ich %d, i_c %d, i_r %d k_c %d k_r %d\n", och, ich, i_c, i_r, k_c, k_r);

		// we compute input and kernel addresses
		int i_addr = i_addr_3 + ((i_r + k_r) * IR_STRIDE);
		int k_addr = k_addr_3 + k_r * KR_STRIDE;

		// we multiply input pixel by kernel element
		out += I[i_addr] * K[k_addr];

              }

	    }

	    // we compute output address
            int o_addr = o_addr_3 + (o_r * OR_STRIDE);

	    // we write the output pixel
	    O[o_addr] = out;

	    // next output row
	    o_r++;

	  }

	  // next output column
	  o_c++;

        }

      }

    }

  }

}


// k_conv2d kernel with subchannels of a given size and with local memory

/*#define SCH_ROWS 16
#define SCH_COLS 16
void k_conv2d(int batch_size,
              float *I, int Irows, int Icols, int Ichannels,
              float *K, int Krows, int Kcols,
              float *B, int use_bias,
              float *O, int Orows, int Ocols, int Ochannels, int Osubchannels_rows, int Osubchannels_cols,
              int padding_rows, int padding_cols,
              int stride_rows, int stride_cols) {

  #pragma HLS INTERFACE m_axi port=I    offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=K    offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B    offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=O    offset=slave bundle=gmem

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
  
  // we sweep all batches
  for (int b=0; b < batch_size; b++) {

    // we sweep all output channels
    for (int och=0; och < Ochannels; och++) {

      // we sweep all subchannels in rows
      for (int osch_r=0; osch_r < Osubchannels_rows; osch_r++) {

	int isch_r = osch_r;

        // we sweep all subchannels in cols
	for (int osch_c=0; osch_c < Osubchannels_cols; osch_c++) {

	  int isch_c = osch_c;

          // We initialize the output grid
	  float output[SCH_ROWS][SCH_COLS];

	  for (int r=0; r < SCH_ROWS; r++) {
	    for (int c=0; c < SCH_COLS; r++) {
	      output[r][c] = 0.f;
	    }
	  }

	  // we sweep all input channels
	  for (int ich=0; ich < Ichannels; ich++) {

            // we read the kernel for the input channel and the output channel
	    float kernel[3][3];
	    for (int r=0; r<3; r++) {
	      for (int c=0; c<3; c++) {
		int addr = ich * KERNEL_ICHANNEL_STRIDE + och * KERNEL_OCHANNEL_STRIDE + r * KERNEL_ROW_STRIDE + c * KERNEL_COL_STRIDE;
	        kernel[r][c] = K[addr];
              }
	    }

	    // now we read all the input subchannel and perform all multiplications
	    for (int r=0; r < SCH_ROWS; r++) {
              for (int c=0; c < SCH_COLS; c++) {
	        int addr = ich * I_CHANNEL_STRIDE + isch_r * I_SCH_ROW_STRIDE + isch_c * I_SCH_COL_STRIDE + r * I_ROW_STRIDE + c * I_COL_STRIDE + b * I_BATCH_STRIDE;
		float A = I[addr];

		for (int kr=0; kr < 2; kr++) {
	          for (int kc=0; kc < 2; kc++) {
		    int row = r + 1 - kr;
		    int col = c + 1 - kc;
		    if ((row != 16) && (col != 16)) output[row][col] += A * kernel[kr][kc];
	          }
		}
	      }
	    }
          }

          // we write the output subchannel
          for (int r=0; r < SCH_ROWS; r++) {
            for (int c=0; c < SCH_COLS; c++) {
	      int addr = och * O_CHANNEL_STRIDE + osch_r * O_SCH_ROW_STRIDE + osch_c * O_SCH_COL_STRIDE + r * O_ROW_STRIDE + c * O_COL_STRIDE + b * O_BATCH_STRIDE;
	      O[addr] = output[r][c];
            }
	  }
        }
      }
    }
  }
}
*/
} // end extern "C"
