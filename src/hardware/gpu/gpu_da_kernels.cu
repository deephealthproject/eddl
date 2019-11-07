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

__global__ void rotate(float* A, float* B, int batch, int depth, int irows, int icols, float angle, int* axis, bool reshape, int mode, float constant){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    // Not implemented
    if (thread_id_x < ops){
        B[thread_id_x] = constant;
    }
}

__global__ void scale(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* new_shape, int mode, float constant){
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
        int Ai = (Bi * irows) / new_shape[0];
        int Aj = (Bj * icols) / new_shape[1];

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
       int offsets[2] = {0, 0};
       offsets[0] = irows/2.0f - orows/2.0f+1;
       offsets[1] = icols/2.0f - ocols/2.0f+1;

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

       // Due to the offsets
       int B_pos = b*B_stride[0] + c*B_stride[1] + (Bi-offsets[0])*B_stride[2] + (Bj-offsets[1])*B_stride[3];
       if (Ai >= coords_from[0] && Ai <= coords_to[0] && Aj >= coords_from[1] && Aj <= coords_to[1]){
           int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
           B[B_pos] = A[A_pos];
       }else{
          B[B_pos] = constant;
      }

       
   }
}


__global__ void crop_scale(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, int* coords_from, int* coords_to, float constant){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    if (thread_id_x < ops){
        int A_wc = coords_to[0]-coords_from[0]+1;
        int A_hc = coords_to[0]-coords_from[1]+1;

        int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
        int B_stride[4] = {depth*orows*ocols, orows*ocols, ocols, 1};

        //--------------
        int b = thread_id_x / B_stride[0] % batch;
        int c = thread_id_x / B_stride[1] % depth;
        int Bi = thread_id_x / B_stride[2] % orows;
        int Bj = thread_id_x / B_stride[3] % ocols;

        // Interpolate indices
        int Ai = (Bi * A_hc) / orows + coords_from[0];
        int Aj = (Bj * A_wc) / ocols + coords_from[1];

        int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
        B[thread_id_x] = A[A_pos];
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


__global__ void shift_random(float* A, float* B, int batch, int depth, int irows, int icols, float* factor_x, float* factor_y, int mode, float constant, float* rnd){
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

        int shift_x = (int)(icols * ((factor_x[1]-factor_x[0]) * rnd[b] + factor_x[0]));
        int shift_y = (int)(irows * ((factor_y[1]-factor_y[0]) * rnd[b+1] + factor_y[0]));

        int Ai = Bi - shift_x;
        int Aj = Bj - shift_y;

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

__global__ void rotate_random(float* A, float* B, int batch, int depth, int irows, int icols, float* factor, int* axis, int mode, float constant, float* rnd){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    long int ops = batch * depth*irows*icols;

    // TODO: Implement
    if (thread_id_x < ops){
        B[thread_id_x] = constant;
    }
}

__global__ void scale_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor, int mode, float constant, float* rnd){
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

        // TODO: Center image
        float scale = (factor[1]-factor[0]) * rnd[b] + factor[0];
        int new_shape_x = (int)(icols * scale);
        int new_shape_y = (int)(irows * scale);
        
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

__global__ void flip_random(float* A, float* B, int batch, int depth, int irows, int icols, int axis, float* rnd){
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

        bool apply = rnd[b] >= 0.5f;
        if (apply){
            int pos[2] = {Bi, Bj};
            if(axis+2==2){ pos[axis] = (irows-1) - pos[axis]; }
            else if(axis+2==3){ pos[axis] = (icols-1) - pos[axis]; }

            int Ai = pos[0];
            int Aj = pos[1];

            if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols){
                int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
                B[thread_id_x] = A[A_pos];
            }
        }else{
            B[thread_id_x] = A[thread_id_x];
        }
        
    }
}


__global__ void crop_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor_x, float* factor_y, float constant, float* rnd){
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

        //--------------
        // printf("A={%d, %d, %d, %d}\n", b, c, Ai, Aj);
        // printf("B={%d, %d, %d, %d}\n", b, c, Bi, Bj);

        // Performs a crop with padding
        int offsets[2] = {0, 0};
        offsets[0] = irows/2.0f - orows/2.0f+1;
        offsets[1] = icols/2.0f - ocols/2.0f+1;

        // Compute random coordinates
        int x1 = (int)(icols * (factor_x[1]-factor_x[0]) * rnd[b] + factor_x[0]);
        int x2 = (int)(icols * (factor_x[1]-factor_x[0]) * rnd[b+1] + factor_x[0]);
        int y1 = (int)(irows * (factor_y[1]-factor_y[0]) * rnd[b+2] + factor_y[0]);
        int y2 = (int)(irows * (factor_y[1]-factor_y[0]) * rnd[b+3] + factor_y[0]);

        int coords_from_x = min(x1, x2);
        int coords_to_x = max(x1, x2);
        int coords_from_y = min(y1, y2);
        int coords_to_y = max(y1, y2);

        int Ai = Bi;  
        int Aj = Bj;
        if(irows!=orows) { Ai+= coords_from_x; }
        if(icols!=ocols) { Aj+= coords_from_y; }


        if (Ai >= coords_from_y && Ai <= coords_to_y && Aj >= coords_from_x && Aj <= coords_to_x){
            int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
            B[thread_id_x] = A[A_pos];
        }else{
            B[thread_id_x] = constant;
        }


    }
}


__global__ void crop_scale_random(float* A, float* B, int batch, int depth, int irows, int icols, int orows, int ocols, float* factor_x, float* factor_y, float constant, float* rnd) {
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

        // Compute random coordinates
        int x1 = (int)(icols * (factor_x[1]-factor_x[0]) * rnd[b] + factor_x[0]);
        int x2 = (int)(icols * (factor_x[1]-factor_x[0]) * rnd[b+1] + factor_x[0]);
        int y1 = (int)(irows * (factor_y[1]-factor_y[0]) * rnd[b+2] + factor_y[0]);
        int y2 = (int)(irows * (factor_y[1]-factor_y[0]) * rnd[b+3] + factor_y[0]);

        int coords_from_x = min(x1, x2);
        int coords_to_x = max(x1, x2);
        int coords_from_y = min(y1, y2);
        int coords_to_y = max(y1, y2);
        
        int A_hc = coords_to_y-coords_from_y+1;
        int A_wc = coords_to_x-coords_from_x+1;

        // Interpolate indices
        int Ai = (Bi * A_hc) / orows + coords_from_x;
        int Aj = (Bj * A_wc) / ocols + coords_from_y;

        int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
        B[thread_id_x] = A[A_pos];
    }
}

__global__ void cutout_random(float* A, float* B, int batch, int depth, int irows, int icols, float* factor_x, float* factor_y, float constant, float* rnd){
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

        // Compute random coordinates
        int x1 = (int)(icols * (factor_x[1]-factor_x[0]) * rnd[b] + factor_x[0]);
        int x2 = (int)(icols * (factor_x[1]-factor_x[0]) * rnd[b+1] + factor_x[0]);
        int y1 = (int)(irows * (factor_y[1]-factor_y[0]) * rnd[b+2] + factor_y[0]);
        int y2 = (int)(irows * (factor_y[1]-factor_y[0]) * rnd[b+3] + factor_y[0]);

        int coords_from_x = min(x1, x2);
        int coords_to_x = max(x1, x2);
        int coords_from_y = min(y1, y2);
        int coords_to_y = max(y1, y2);
        
        if (Bi >= coords_from_y && Bi <= coords_to_y && Bj >= coords_from_x && Bj <= coords_to_x){
            B[thread_id_x] = constant;
        }else{
            B[thread_id_x] = A[thread_id_x];
        }

    }
}