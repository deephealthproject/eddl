#include <math.h>
#include <stdio.h>
extern "C" {

//#define DEBUG

#define BLOCK_SIZE 64

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

  // Local storage for a block of input matrices A and B and C
  float A_local[BLOCK_SIZE][BLOCK_SIZE];
  float B_local[BLOCK_SIZE][BLOCK_SIZE];
  float C_local[BLOCK_SIZE][BLOCK_SIZE];

  #ifdef DEBUG
  printf("Matmul2D (fpga): A[%d x %d] x B[%dx%d] tA %d tB %d incC %d\n", Ashape0, Ashape1, Bshape0, Bshape1, tA, tB, incC);
  #endif

  // real rows and cols of each matrix
  int Acols = Ashape1;
  int Arows = Ashape0;
  int Bcols = Bshape1;
  int Brows = Bshape0;
  int Crows;
  int Ccols;
  if (!tA) Crows = Ashape0; else Crows = Ashape1;
  if (!tB) Ccols = Bshape1; else Ccols = Bshape0;

  #ifdef DEBUG
  printf("Acols %d Arows %d Bcols %d Brows %d Ccols %d Crows %d\n", Acols, Arows, Bcols, Brows, Ccols, Crows);
  #endif

  // extra rows and cols
  int A_extra_rows = Arows % BLOCK_SIZE;
  int A_extra_cols = Acols % BLOCK_SIZE;
  int B_extra_rows = Brows % BLOCK_SIZE;
  int B_extra_cols = Bcols % BLOCK_SIZE;
  int C_extra_rows = Crows % BLOCK_SIZE;
  int C_extra_cols = Ccols % BLOCK_SIZE;

  #ifdef DEBUG
  printf("Extra: Arows %d Acols %d Brows %d Bcols %d Crows %d Ccols %d\n", A_extra_rows, A_extra_cols, B_extra_rows, B_extra_cols, C_extra_rows, C_extra_cols);
  #endif

  // C blocks
  int C_block_cols = (Ccols / BLOCK_SIZE) + (C_extra_cols != 0);
  int C_block_rows = (Crows / BLOCK_SIZE) + (C_extra_rows != 0);

  #ifdef DEBUG
  printf("C_blocks: rows %d cols %d\n", C_block_rows, C_block_cols);
  #endif

  // number of blocks to read from A and B for each C block
  int num_blocks;
  if (!tA) num_blocks = (Acols / BLOCK_SIZE) + (A_extra_cols != 0); else num_blocks = (Arows / BLOCK_SIZE) + (A_extra_rows != 0);

  #ifdef DEBUG
  printf("num_blocks %d\n", num_blocks);
  #endif

  // we sweep all C blocks
  for (int cbr = 0; cbr < C_block_rows; cbr++) {

    #ifdef DEBUG
    printf("C block row %d\n", cbr);
    #endif

    // we set the last row
    int c_last_r = (cbr == C_block_rows-1) ? C_extra_rows : BLOCK_SIZE;


    #ifdef DEBUG
    printf("  - c_last_r %d\n", c_last_r);
    #endif

    for (int cbc = 0; cbc < C_block_cols; cbc++) {

      #ifdef DEBUG
      printf("C block col %d\n", cbc);
      #endif

      // C block address
      int C_block_addr = (cbr * BLOCK_SIZE * Ccols) + (cbc * BLOCK_SIZE);

  	// we set the last col
      int c_last_c = (cbc == C_block_cols-1) ? C_extra_cols : BLOCK_SIZE;

      #ifdef DEBUG
      printf("  - c_last_c %d\n", c_last_c);
      #endif

      // we set to zero C local block
      for (int c=0; c<BLOCK_SIZE; c++) {
        for (int r=0; r<BLOCK_SIZE; r++) {
          C_local[r][c] = 0.f;
        }
      }

      #ifdef DEBUG
      printf("  - C_local initialized to zero\n");
      #endif

      // we read C block if needed
      if (incC) {
        for (int c=0; c<c_last_c; c++) {
          for (int r=0; r<c_last_r; r++) {
            int C_addr = C_block_addr + (r * Ccols) + c;
	    C_local[r][c] = C[C_addr];
          }
        }
      }

      #ifdef DEBUG
      printf("  - C_matrix read (or bypassed)\n");
      #endif

      // we sweep all involved blocks from A and B
      for (int b=0; b<num_blocks; b++) {

        #ifdef DEBUG
	printf("block (compute) %d\n", b);
        #endif

        // we read A block
        int A_block_addr;
        if (!tA) A_block_addr = (cbr * BLOCK_SIZE * Acols) + (b * BLOCK_SIZE); else A_block_addr = (cbc * BLOCK_SIZE) + (b * BLOCK_SIZE * Acols);

        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
            A_local[r][c] = 0.f;
          }
        }

        #ifdef DEBUG
	printf("A_local reset\n");
        #endif

        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
	    int a_col = !tA ? (b * BLOCK_SIZE + c) : (cbc * BLOCK_SIZE + r); 
	    int a_row = !tA ? (cbr * BLOCK_SIZE + r) : (b * BLOCK_SIZE + c);
//	    printf("A_load: row %d col %d -> a_row %d a_col %d\n", r, c, a_row, a_col);
	    if ((a_col < Acols) && (a_row < Arows)) {
              int A_addr;
              if (!tA) A_addr = A_block_addr + (r * Acols) + c; else A_addr = A_block_addr + (c * Acols) + r;
//	      printf(" - A_addr %d\n", A_addr);
              A_local[r][c] = A[A_addr];
	    }
          }
        }

        #ifdef DEBUG
	printf("A_local loaded\n");
        #endif
        // we read B block
        int B_block_addr;
        if (!tB) B_block_addr = (cbc * BLOCK_SIZE) + (b * BLOCK_SIZE * Bcols); else B_block_addr = (cbr * BLOCK_SIZE * Bcols) + (b * BLOCK_SIZE);

        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
            B_local[r][c] = 0.f;
          }
        }

        #ifdef DEBUG
	printf("B_local reset\n");
        #endif

        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
	    int b_col = !tB ? (cbc * BLOCK_SIZE + c) : (b * BLOCK_SIZE + r);
	    int b_row = !tB ? (b * BLOCK_SIZE + r) : (cbc * BLOCK_SIZE + c);
//	    printf("B_load: row %d col %d -> b_row %d b_col %d\n", r, c, b_row, b_col);
	    if ((b_col < Bcols) && (b_row < Brows)) {
              int B_addr;
              if (!tB) B_addr = B_block_addr + (r * Bcols) + c; else B_addr = B_block_addr + (c * Bcols) + r;
//	      printf(" - B_addr %d\n", B_addr);
              B_local[r][c] = B[B_addr];
	    }
          }
        }

        #ifdef DEBUG
	printf("B_local loaded\n");
        #endif

        // We compute now A_local * B_local
        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
            float v = 0.f;
            for (int k=0; k<BLOCK_SIZE; k++) {
              v = v + (A_local[r][k] * B_local[k][c]);
            }
            C_local[r][c] = C_local[r][c] + v;
          }
        }

        #ifdef DEBUG
	printf("C_local computed and accumulated\n");
        #endif
      }

      // We write the C block
      for (int c=0; c<c_last_c; c++) {
        for (int r=0; r<c_last_r; r++) {
          int C_addr = C_block_addr + (r * Ccols) + c;
          C[C_addr] = C_local[r][c];
        } 
      } 

      #ifdef DEBUG
      printf("C block written\n");
      #endif
    } 
  } 
} 

}
