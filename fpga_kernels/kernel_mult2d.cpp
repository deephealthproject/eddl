#include <math.h>
#include <stdio.h>
extern "C" {

void k_mult2d(float *A, float *B, float *C, int Ashape0, int Ashape1, int Bshape0, int Bshape1, int tA, int tB, int incC){

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=C  bundle=control
#pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
#pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
#pragma HLS INTERFACE s_axilite port=Bshape0 bundle=control
#pragma HLS INTERFACE s_axilite port=Bshape1 bundle=control
#pragma HLS INTERFACE s_axilite port=tA bundle=control
#pragma HLS INTERFACE s_axilite port=tB bundle=control


    if (!tA && !tB) {
      int ar = Ashape0;
      int ac = Ashape1;
      int br = Bshape0;
      int bc = Bshape1;
      int cr = Ashape0;
      int cc = Bshape1;

      for (int r = 0; r < ar; r++) {
        for (int c = 0; c < bc; c++) {
          for (int k = 0; k < ac; k++) {
	    int c_addr = (r * cc) + c;
	    int a_addr = (r * ac) + k;
	    int b_addr = (k * bc) + r;
	    if (incC) C[c_addr] += A[a_addr] * B[b_addr];
	    else C[c_addr] = A[a_addr] * B[b_addr];
	  }
	}
      }
    }

    if (!tA && tB) {
      int ar = Ashape0;
      int ac = Ashape1;
      int br = Bshape1;
      int bc = Bshape0;
      int cr = Ashape0;
      int cc = Bshape0;
      
      for (int r = 0; r < ar; r++) { 
        for (int c = 0; c < bc; c++) { 
          for (int k = 0; k < ac; k++) {
            int c_addr = (r * cc) + c;
            int a_addr = (r * ac) + k;
            int b_addr = (r * bc) + k;
            if (incC) C[c_addr] += A[a_addr] * B[b_addr];
            else C[c_addr] = A[a_addr] * B[b_addr];
          }     
        }       
      }       
    }     
 
    if (tA && !tB) {
      int ar = Ashape1;
      int ac = Ashape0;
      int br = Bshape0;
      int bc = Bshape1;
      int cr = Ashape1;
      int cc = Bshape1;
      
      for (int r = 0; r < ar; r++) { 
        for (int c = 0; c < bc; c++) { 
          for (int k = 0; k < ac; k++) {
            int c_addr = (r * cc) + c;
            int a_addr = (k * ac) + r;
            int b_addr = (k * bc) + r;
            if (incC) C[c_addr] += A[a_addr] * B[b_addr];
            else C[c_addr] = A[a_addr] * B[b_addr];
          }     
        }       
      }       
    }     

    if (tA && tB) {
      int ar = Ashape1;
      int ac = Ashape0;
      int br = Bshape1;
      int bc = Bshape0;
      int cr = Ashape1;
      int cc = Bshape0;

      for (int r = 0; r < ar; r++) {
        for (int c = 0; c < bc; c++) {
          for (int k = 0; k < ac; k++) {
            int c_addr = (r * cc) + c;
            int a_addr = (k * ac) + r;
            int b_addr = (r * bc) + k;
            if (incC) C[c_addr] += A[a_addr] * B[b_addr];
            else C[c_addr] = A[a_addr] * B[b_addr];
          }
        }
      }
    }
  }
}


// void cpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
//     _profile(_CPU_MULT2D, 0);
//   if (!tB) {
//     if (!tA) {
//       if (!incC) *(C->ptr2) = *(B->ptr2) * (*(A->ptr2));
//       else *(C->ptr2) += *(B->ptr2) * (*(A->ptr2));
//     } else {
//       if (!incC) *(C->ptr2) = *(B->ptr2) * ((*(A->ptr2)).transpose());
//       else *(C->ptr2) += *(B->ptr2) * ((*(A->ptr2)).transpose());
//     }
//   } else {
//     if (!tA) {
//       if (!incC) *(C->ptr2) = (*(B->ptr2)).transpose() * (*(A->ptr2));
//       else *(C->ptr2) += (*(B->ptr2)).transpose() * (*(A->ptr2));
//     } else {
//       if (!incC) *(C->ptr2) = (*(B->ptr2)).transpose() * ((*(A->ptr2)).transpose());
//       else *(C->ptr2) += (*(B->ptr2)).transpose() * ((*(A->ptr2)).transpose());
//     }
//   }
//     _profile(_CPU_MULT2D, 1);
// }
