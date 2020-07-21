#include <math.h>
#include <stdio.h>
extern "C" {

//#define DEBUG
// JFLICH
#define BLOCK_SIZE 64 

// JM10  
//#define BLOCK_SIZE 4


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
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Local storage for a block of input matrices A and B and C
  float A_local[BLOCK_SIZE][BLOCK_SIZE];
  float B_local[BLOCK_SIZE][BLOCK_SIZE];
  float C_local[BLOCK_SIZE][BLOCK_SIZE];

  #ifndef __SYNTHESIS__
  printf("mult2d (fpga): A[%d,%d] x B[%d,%d] tA %d tB %d incC %d\n", Ashape0, Ashape1, Bshape0, Bshape1, tA, tB, incC);
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

  #ifndef __SYNTHESIS__
  printf("Arows %d  Acols %d  Brows %d  Bcols %d  -->>  Crows %d  Ccols %d \n", Arows, Acols, Brows, Bcols, Crows, Ccols);
  #endif

  // extra rows and cols
  int A_extra_rows = Arows % BLOCK_SIZE;
  int A_extra_cols = Acols % BLOCK_SIZE;
  int B_extra_rows = Brows % BLOCK_SIZE;
  int B_extra_cols = Bcols % BLOCK_SIZE;
  int C_extra_rows = Crows % BLOCK_SIZE;
  int C_extra_cols = Ccols % BLOCK_SIZE;

  #ifdef DEBUG
  #ifndef __SYNTHESIS__
  printf("Extra: Arows %d Acols %d Brows %d Bcols %d Crows %d Ccols %d\n",
      A_extra_rows, A_extra_cols, B_extra_rows, B_extra_cols, C_extra_rows, C_extra_cols);
  #endif
  #endif

  // C blocks
  int C_block_cols = (Ccols / BLOCK_SIZE) + (C_extra_cols != 0);
  int C_block_rows = (Crows / BLOCK_SIZE) + (C_extra_rows != 0);

  #ifdef DEBUG
  #ifndef __SYNTHESIS__
  printf("C_blocks: rows %d cols %d\n", C_block_rows, C_block_cols);
  #endif
  #endif

  // number of blocks to read from A and B for each C block
  int num_blocks;
  if (!tA) num_blocks = (Acols / BLOCK_SIZE) + (A_extra_cols != 0); 
  else     num_blocks = (Arows / BLOCK_SIZE) + (A_extra_rows != 0);

  #ifdef DEBUG
  #ifndef __SYNTHESIS__
  printf("num_blocks %d\n", num_blocks);
  #endif
  #endif

  // we sweep all C blocks
  for (int cbr = 0; cbr < C_block_rows; cbr++) {
    // we set the last row to read for current C_block
    //int c_last_r = (cbr == C_block_rows-1) ? C_extra_rows : BLOCK_SIZE;
    int c_last_r = BLOCK_SIZE;
    if (cbr == (C_block_rows -1)) {
      if ((Crows % BLOCK_SIZE) != 0) {
        c_last_r = C_extra_rows;
      }
    }

    //#ifdef DEBUG
    //#ifndef __SYNTHESIS__
    //printf("C_block row %3d   c_last_r = %3d    \n", cbr, c_last_r);
    //#endif
    //#endif

    for (int cbc = 0; cbc < C_block_cols; cbc++) {
      // C block address
      int C_block_addr = (cbr * BLOCK_SIZE * Ccols) + (cbc * BLOCK_SIZE);
     
      // we set the last col
      // int c_last_c = (cbc == C_block_cols-1) ? C_extra_cols : BLOCK_SIZE;
      int c_last_c = BLOCK_SIZE;
      if (cbc == (C_block_cols -1)) {
        if ((Ccols % BLOCK_SIZE) != 0) {
          c_last_c = C_extra_cols;
        }
      }
      #ifdef DEBUG
      #ifndef __SYNTHESIS__
      printf("C_block_row %3d (c_last_r %3d)   C_block_col %3d (c_last_c %3d)   C_block_addr %6d\n", cbr, c_last_r, cbc, c_last_c, C_block_addr);
      #endif
      #endif

      // we set to zero C local block
      #ifdef DEBUG
      #ifndef __SYNTHESIS__
      printf("Initialize C_local to zero\n");
      #endif
      #endif


      for (int c=0; c<BLOCK_SIZE; c++) {
        for (int r=0; r<BLOCK_SIZE; r++) {
          C_local[r][c] = 0.f;
        }
      }

      // we read C block if needed
      if (incC) {
        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("incC detected, read C_matrix\n");
        #endif
        #endif
        for (int c=0; c<c_last_c; c++) {
          for (int r=0; r<c_last_r; r++) {
            int C_addr = C_block_addr + (r * Ccols) + c;
	          C_local[r][c] = C[C_addr];
          }
        }
      } 
      //else {
      //  #ifdef DEBUG
      //  #ifndef __SYNTHESIS__
      //  printf("NON incremental matrix multiplication, C = AxB\n");
      //  #endif
      //  #endif
      //}

      // we sweep all involved blocks from A and B
      #ifdef DEBUG
      #ifndef __SYNTHESIS__
      printf("let's sweep all involved blocks (%2d) from A and B to calculate C_block [%3d,%3d]\n", 
          num_blocks, cbr, cbc);
      #endif
      #endif

      for (int b=0; b<num_blocks; b++) {
        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("C_block [%3d,%3d] , compute A-B matrices block index %2d (%2d of %2d)\n", 
            cbr, cbc, b, b+1, num_blocks);
        #endif
        #endif

        // we read A block
        int A_block_addr;
        if (!tA) {
          A_block_addr = (cbr * BLOCK_SIZE * Acols) + (b * BLOCK_SIZE); 
        } else {
          A_block_addr = (cbr * BLOCK_SIZE)         + (b * BLOCK_SIZE * Acols);
        }

        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("block (compute) %d  A_block_addr (%4d)\n", b, A_block_addr);
        #endif
        #endif
        
        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
            A_local[r][c] = 0.f;
          }
        }
        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("A_local reset\n");
        #endif
        #endif
        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
            int a_col = !tA ? (b * BLOCK_SIZE + c) : (cbc * BLOCK_SIZE + r); 
            int a_row = !tA ? (cbr * BLOCK_SIZE + r) : (b * BLOCK_SIZE + c);

            //#ifdef DEBUG
            //#ifndef __SYNTHESIS__
            //printf("  a_row (%4d)  a_col (%4d)\n", a_row, a_col);
            //#endif
            //#endif

            //if ((a_col < Acols) && (a_row < Arows)) {
              int A_addr;
              if (!tA) A_addr = A_block_addr + (r * Acols) + c; 
              else     A_addr = A_block_addr + (c * Acols) + r;

            if ( (tA && (A_addr < (Arows*Acols))) || ((!tA) && ((a_col < Acols) && (a_row < Arows)) ) ){
              #ifdef DEBUG
              #ifndef __SYNTHESIS__
              printf(" - A_local[%d][%d] = A[%d] = %2.2f \n", r,c, A_addr, A[A_addr]);
              #endif
              #endif

              A_local[r][c] = A[A_addr];
            }
          }
        }

        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("A_local loaded\n");
        #endif
        #endif

        // we read B block
        int B_block_addr;
        if (!tB) B_block_addr = (cbc * BLOCK_SIZE)         + (b * BLOCK_SIZE * Bcols); 
        else     B_block_addr = (cbc * BLOCK_SIZE * Bcols) + (b * BLOCK_SIZE);
       
        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("block (compute) %d  B_block_addr (%4d)\n", b, B_block_addr);
        #endif
        #endif

        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
            B_local[r][c] = 0.f;
          }
        }
        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("B_local reset\n");
        #endif
        #endif

        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
            int b_col = !tB ? (cbc * BLOCK_SIZE + c) : (b * BLOCK_SIZE + r);
            int b_row = !tB ? (b * BLOCK_SIZE + r) : (cbc * BLOCK_SIZE + c);

            //if ((b_col < Bcols) && (b_row < Brows)) {
              int B_addr;
              if (!tB) B_addr = B_block_addr + (r * Bcols) + c; 
              else     B_addr = B_block_addr + (c * Bcols) + r;

            if ( (tB && (B_addr < (Brows*Bcols))) || ((!tB) && ((b_col < Bcols) && (b_row < Brows)))) {
              #ifdef DEBUG
              #ifndef __SYNTHESIS__
              printf(" - B_local[%d][%d] = B[%d] = %2.2f \n", r,c, B_addr, B[B_addr]);
              #endif
              #endif

              B_local[r][c] = B[B_addr];
            }
          }
        }


        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("B_local loaded\n");
        #endif
        #endif
   
/*      
        // JFLICH
        // We compute now A_local * B_local
        for (int c=0; c<BLOCK_SIZE; c++) {
          for (int r=0; r<BLOCK_SIZE; r++) {
            float v = 0.f;
            for (int k=0; k<BLOCK_SIZE; k++) {
              v = v + (A_local[r][k] * B_local[k][c]);
            }
            C_local[r][c] = C_local[r][c] + v;
#ifdef DEBUG
#ifndef __SYNTHESIS__
            printf("  C_local[%d][%d] = %2.2f\n", r, c, C_local[r][c] );
#endif
#endif
          }
        }
*/      


        // JM10
        
        // We compute now A_local * B_local
        for (int r=0; r<BLOCK_SIZE; r++) {
          for (int c=0; c<BLOCK_SIZE; c++) {
            #ifdef DEBUG
            #ifndef __SYNTHESIS__
            printf(" Calculate C_local[%d][%d]\n", r, c);
            #endif
            #endif
            float v = 0.f;
            for (int k=0; k<BLOCK_SIZE; k++) {
              v = v + (A_local[r][k] * B_local[k][c]);
              #ifdef DEBUG
              #ifndef __SYNTHESIS__
              printf("   C_local[%d][%d] = C_local[%d][%d] + A[%d][%d] * B[%d][%d] \n", r,c,r,c,r,k,k,c);
              #endif
              #endif
            }
            C_local[r][c] = C_local[r][c] + v;
            #ifdef DEBUG
            #ifndef __SYNTHESIS__
            printf("  C_local[%d][%d] = %2.2f\n", r, c, C_local[r][c] );
            #endif
            #endif
          }
        }

        #ifdef DEBUG
        #ifndef __SYNTHESIS__
        printf("C_local computed and accumulated\n");
        #endif
        #endif
      }

      // We write the C block
      for (int c=0; c<c_last_c; c++) {
        for (int r=0; r<c_last_r; r++) {
          int C_addr = C_block_addr + (r * Ccols) + c;
          C[C_addr] = C_local[r][c];

          #ifdef DEBUG
          #ifndef __SYNTHESIS__
          printf("C[%d] = C_local[%d][%d] = %2.2f\n", C_addr, r, c, C_local[r][c]);
          #endif
          #endif
        } 
      } 

      #ifdef DEBUG
      #ifndef __SYNTHESIS__
      printf("C block written\n\n");
      #endif
      #endif
    } 
  
  
  }
  
  #ifndef __SYNTHESIS__
  printf("C block written\n\n");
  #endif

} // end kernel 

} // end extern "C"
