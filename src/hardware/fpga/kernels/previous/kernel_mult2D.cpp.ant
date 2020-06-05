/**********

*******************************************************************************/
#include <stdio.h>
#include <math.h>

extern "C" {

void k_mult2D(
         const float *A, 
         const float *B, 
         float *C,
         int a_row, 
	 int a_col, 
         int b_row,
         int b_col, 
         int tA, 
         int tB
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=C  bundle=control
#pragma HLS INTERFACE s_axilite port=a_row bundle=control
#pragma HLS INTERFACE s_axilite port=a_col bundle=control
#pragma HLS INTERFACE s_axilite port=b_row bundle=control
#pragma HLS INTERFACE s_axilite port=b_col bundle=control
#pragma HLS INTERFACE s_axilite port=tA bundle=control
#pragma HLS INTERFACE s_axilite port=tB bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    // Matrix A is nxy and B is yxm --> size    
    // locat variables to bring data to FPGA
   
    int c_row;//a_row;
    int c_col;//b_col;  
    int ar;
    int ac;
    int br; 
    int bc;

    if (!tA){
       c_row = a_row;
       ar = a_row;       
       ac = a_col;
       if (!tB) {c_col = b_col; br = b_row; bc = b_col; } else {c_col = b_row; br=b_col; bc=b_row;}
    } else {
       ar = a_col;
       ac = a_row;
       c_row = a_col;
       if (!tB) {c_col = b_col; br = b_row; bc = b_col; } else {c_col = b_row; br=b_col; bc=b_row;}
    }

    if (!tA & !tB) { 
       for (int i = 0; i < ar; i++) {
          for (int j = 0; j < bc; j++) {
             for (int k = 0; k < br; k++)
                C[i*c_col+j]+=A[i*ac+k]*B[k*bc+j];
          }
       }
    }
    
    if (!tA & tB) {
       for (int i = 0; i < br; i++) {
          for (int j = 0; j < ac; j++) {
             for (int k = 0; k < ar; k++)
                C[i*c_col+j]+=A[i*ac+k]*B[k*bc+j];
          }
       }
    }

    if (tA & !tB) {
       for (int i = 0; i < br; i++) {
          for (int j = 0; j < ac; j++) {
             for (int k = 0; k < br; k++)
                C[i*c_col+j]+=B[i*bc+k]*A[k*ac+j];
          }
       }
    }
    
    if (tA & tB) {
       for (int i = 0; i < ar; i++) {
          for (int j = 0; j < bc; j++) {
             for (int k = 0; k < br; k++)
                C[j*c_col+i]+=A[i*ac+k]*B[k*bc+j];
          }
       }
    }



}
} 
