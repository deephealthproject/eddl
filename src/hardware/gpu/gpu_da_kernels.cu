/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_kernels.h"


__global__ void shift(float* A, float* B, int batch, int depth, int irows, int icols, int* shift, int mode, float constant){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
        int *B_stride = A_stride;

        //--------------
        int b = thread_id_x / B_stride[0] % batch;
        int c = thread_id_x / B_stride[1] % depth;
        int Bi = thread_id_x / B_stride[2] % irows;
        int Bj = thread_id_x / B_stride[3] % icols;
        //--------------
        //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);

        int Ai = Bi - shift[0];
        int Aj = Bj - shift[1];

        if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols){
            int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
            B[thread_id_x] = A[A_pos];
        }else{
            if(mode==0){ // constant
                B[thread_id_x] = constant;
            }
        }
    }

}


__global__ void scale(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int mode, float constant){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*orows*ocols;

    if (thread_id_x < ops){
        int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
        int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

        //--------------
        int b = thread_id_x / B_stride[0] % batch;
        int c = thread_id_x / B_stride[1] % depth;
        int Bi = thread_id_x / B_stride[2] % orows;
        int Bj = thread_id_x / B_stride[3] % ocols;
        //--------------
        //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);

        // Interpolate indices
        int Ai = (Bi * irows) / orows;
        int Aj = (Bj * icols) / ocols;

        if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols){
            int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
            B[thread_id_x] = A[A_pos];
        }else{
            if(mode==0){ // constant
                B[thread_id_x] = constant;
            }
        }
    }

}

__global__ void flip(float* A, float* B, int batch, int depth, int irows, int icols, int axis){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
        int *B_stride = A_stride;

        //--------------
        int b = thread_id_x / B_stride[0] % batch;
        int c = thread_id_x / B_stride[1] % depth;
        int Bi = thread_id_x / B_stride[2] % irows;
        int Bj = thread_id_x / B_stride[3] % icols;
        //--------------
        //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);

        int pos[2] = {Bi, Bj}; 
        if(axis+2==2){ pos[axis] = (irows-1) - pos[axis]; }
        else if(axis+2==3){ pos[axis] = (icols-1) - pos[axis]; }

        int Ai = pos[0]; 
        int Aj = pos[1];

        if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols){
            int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
            B[thread_id_x] = A[A_pos];
        }
    }
}


__global__ void crop(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, float constant){
   long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
   long int ops = batch * depth*irows*icols;

   if (thread_id_x < ops){

       int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
       int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

       //--------------
       int b = thread_id_x / B_stride[0] % batch;
       int c = thread_id_x / B_stride[1] % depth;
       int Bi = thread_id_x / B_stride[2] % orows;
       int Bj = thread_id_x / B_stride[3] % ocols;

       int Ai = Bi;
       int Aj = Bj;
       if(irows!=orows) { Ai+= coords_from[0]; }
       if(icols!=ocols) { Aj+= coords_from[1]; }
       //--------------
       // printf("A={%d, %d, %d, %d}\n", b, c, Ai, Aj);
       // printf("B={%d, %d, %d, %d}\n", b, c, Bi, Bj);


       if (Ai >= coords_from[0] && Ai <= coords_to[0] && Aj >= coords_from[1] && Aj <= coords_to[1]){
           int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
           B[thread_id_x] = A[A_pos];
       }else{
          B[thread_id_x] = constant;
      }

       
   }
}

__global__ void cutout(float* A, float* B, int batch, int depth, int irows, int icols, int* coords_from, int* coords_to, float constant){
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
  long int ops = batch*depth*irows*icols;

  if (thread_id_x < ops){
      int B_stride[4] = {depth*irows*icols, irows*icols, icols, 1};

      //--------------
      int b = thread_id_x / B_stride[0] % batch;
      int c = thread_id_x / B_stride[1] % depth;
      int Bi = thread_id_x / B_stride[2] % irows;
      int Bj = thread_id_x / B_stride[3] % icols;
      //--------------
       //printf("{%d, %d, %d, %d}\n", b, c, Bi, Bj);

      if (Bi >= coords_from[0] && Bi <= coords_to[0] && Bj >= coords_from[1] && Bj <= coords_to[1]){
          B[thread_id_x] = constant;
      }else{
          B[thread_id_x] = A[thread_id_x];
      }

  }
}
