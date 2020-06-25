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

    // Matrix A is nxy and B is yxm --> size
    // locat variables to bring data to FPGA

    int c_row;//a_row;
    int c_col;//b_col;
    int ar;
    int ac;
    int br;
    int bc;

    if (!tA){
       c_row = Ashape0;
       ar = Ashape0;
       ac = Ashape1;
       if (!tB) {c_col = Bshape1; br = Bshape0; bc = Bshape1; } else {c_col = Bshape0; br=Bshape1; bc=Bshape0;}
    } else {
       ar = Ashape1;
       ac = Ashape0;
       c_row = Ashape1;
       if (!tB) {c_col = Bshape1; br = Bshape0; bc = Bshape1; } else {c_col = Bshape0; br=Bshape1; bc=Bshape0;}
    }

    if (!tA & !tB) {
       for (int i = 0; i < ar; i++) {
          for (int j = 0; j < bc; j++) {
             for (int k = 0; k < br; k++){
               if(!incC){
                C[i*c_col+j]=A[i*ac+k]*B[k*bc+j];
                }
                else{C[i*c_col+j]+=A[i*ac+k]*B[k*bc+j];}
              }
          }
       }
    }

    if (!tA & tB) {
       for (int i = 0; i < br; i++) {
          for (int j = 0; j < ac; j++) {
             for (int k = 0; k < ar; k++){
               if(!incC){
                C[i*c_col+j]=A[i*ac+k]*B[k*bc+j];
                }
                else{C[i*c_col+j]+=A[i*ac+k]*B[k*bc+j];}
              }
          }
       }
    }

    if (tA & !tB) {
       for (int i = 0; i < br; i++) {
          for (int j = 0; j < ac; j++) {
             for (int k = 0; k < br; k++){
               if(!incC){
                C[i*c_col+j]=B[i*bc+k]*A[k*ac+j];
                }
                else{C[i*c_col+j]+=B[i*bc+k]*A[k*ac+j];}
              }
          }
       }
    }

    if (tA & tB) {
       for (int i = 0; i < ar; i++) {
          for (int j = 0; j < bc; j++) {
             for (int k = 0; k < br; k++){
               if(!incC){
                C[j*c_col+i]=A[i*ac+k]*B[k*bc+j];
                }
                else{C[j*c_col+i]+=A[i*ac+k]*B[k*bc+j];}
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
