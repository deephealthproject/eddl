#include <math.h>
#include <stdio.h>
extern "C" {


#define BUFFER_SIZE 8

void k_reduce_sum2d(float *A, float *B, int Ashape0, int Ashape1, int axis, int incB) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=axis bundle=control
  #pragma HLS INTERFACE s_axilite port=incB bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  //Local variables
  float Local_A[BUFFER_SIZE][BUFFER_SIZE];
  float Local_B[BUFFER_SIZE];
  int block_row = ceil(float(Ashape0)/BUFFER_SIZE);
  int block_col = ceil(float(Ashape1)/BUFFER_SIZE);
  int Bshape;


  //Checking variables
  if(!incB){
    for(int i = 0; i<Ashape1; i++){
      B[i]=0.0;
    }
  }

  if(axis){
    Bshape = Ashape0;
  }
  else{
    Bshape = Ashape1;
  }

  // Loop tiling
  for(int block_i = 0; block_i < block_row; block_i++ ){
    for(int block_j = 0; block_j < block_col; block_j++){

      // Read A and init Local_A
      for(int i = 0; i < BUFFER_SIZE; i++){
        for(int j = 0; j < BUFFER_SIZE; j++){

          int ii = i + (block_i*BUFFER_SIZE); // Absolute index i
          int jj = j + (block_j*BUFFER_SIZE); // Absolute index j

          // Checking matrix limits
          if((ii >= Ashape0) || (jj >= Ashape1)){
            Local_A[i][j] = 0;
          }
          else{
            Local_A[i][j] = A[jj + Ashape1*ii];
          }

        }
      }

      //Init Local_B
      for(int j=0; j<BUFFER_SIZE; j++){
        Local_B[j] = 0.0;
      }

      //Compute
      for(int i=0;i<BUFFER_SIZE;i++){
        for(int j=0;j<+BUFFER_SIZE;j++){

          if(axis){
            Local_B[i] += Local_A[i][j];
          }
          else{
            Local_B[j] += Local_A[i][j];
          }

        }
      }

      // Write B from Local_B
      for(int j=0; j< BUFFER_SIZE; j++){
        if(axis){
          if(((block_i*BUFFER_SIZE) + j) < Bshape){
            B[(j+(block_i*BUFFER_SIZE))] += Local_B[j];
            }
        }
        else{
          if(((block_j*BUFFER_SIZE) + j) < Bshape){
            B[(j+(block_j*BUFFER_SIZE))] += Local_B[j];
          }
        }

      }


    }
  } //End loop tiling

}
} // end extern "C"
