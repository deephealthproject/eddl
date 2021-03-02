/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"
#include "eddl/hardware/gpu/gpu_kernels.h"
//
//__device__ float get_pixel(int b,int px,int py,int pz, int ircd, int irc, int irows, int icols, float* I){
//    if (px < 0) return 0.0;
//    else if (py < 0) return 0.0;
//    else if (px >= icols) return 0.0;
//    else if (py >= irows) return 0.0;
//    else {
//        int ptr = (b * ircd) + (pz * irc) + (py * icols) + px;
//        return I[ptr];
//    }
//}
//
//__device__ void add_pixel(int b,int px,int py,int pz, int ircd, int irc, int irows, int icols, float* ID, float val) {
//    if (px < 0) return;
//    else if (py < 0) return;
//    else if (px >= icols) return;
//    else if (py >= irows) return;
//    else {
//        int ptr = (b * ircd) + (pz * irc) + (py * icols) + px;
//        ID[ptr] += val;
//    }
//}


// MaxPool
__global__ void maxpool2d(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt, int padrb,int padcl, int padcr,float* indX, float* indY) {

    long int ops = batch * orows * ocols * odepth;
    long int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_id_x < ops) {
        // Parse "thread_id_x" to output(b, d, r, c) ****************
        // output pixel at batch=ob, coord=(or,oc) at map=oz
        int orcd=orows*ocols*odepth; // out size of batch
        int orc=orows*ocols;  // out size of slice
        int ob=thread_id_x/orcd; // batch's index => B[i] || (ib=ob)
        int bm=thread_id_x%orcd; // index inside batch i => thread_id_x=23, batch_size=20: index = 3
        int ouz=bm/orc; // depth index (iuz=ouz)
        int our=(bm%orc)/ocols; // row index
        int ouc=(bm%orc)%ocols; // col index

        // Parse output(b, d, r, c) to input(b, d, r, c) ****************
        int inr = our * sr;  // input row index (without padding)
        int inc = ouc * sc;  // input col index (without padding)
        int ircd=irows*icols*idepth; // in size of batch
        int irc=irows*icols;  // in size of batch

        int min_i = -padrt;
        int max_i = irows+padrb-kr;
        int i = min_i + inr;  // input row index (with padding)

        int min_j = -padcl;
        int max_j = icols+padcr-kc;
        int j = min_j + inc;  // input column index (with padding)

        int b = ob;  // batch
        int k = ouz;  // depth
        int p = thread_id_x;  // index
        /*printf("%d\n", p);*/

        // Check bounds
        if (i <= max_i && j <= max_j){

            float max = GPU_LOWEST_FLOAT;
            //float max = I[i,j];
            for (int ki = 0; ki < kr; ki++){  // rows (kernel): top-bottom
                for (int kj = 0; kj < kc; kj++) {  // cols (kernel): left-right

                    // Get value W[ki,kj] value in window
                    int px = j + kj;
                    int py = i + ki;
                    int pz = k;
                    float v = 0.0;

                    // Get values
                    if (px < 0) v = 0.0;
                    else if (py < 0) v = 0.0;
                    else if (px >= icols) v = 0.0;
                    else if (py >= irows) v = 0.0;
                    else {
                        int ptr = (b * ircd) + (pz * irc) + (py * icols) + px;
                        v = I[ptr];
                    }

                    if (v > max) {
                        max = v;
                        indX[p] = px;
                        indY[p] = py;
                    }
                }// kernel cols
            }// kernel rows

            // Set output value
            O[p] = max;
        }
    }

}

__global__ void maxpool2d_back(float* D, float* ID, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt, int padrb,int padcl, int padcr,float* indX, float* indY) {

    long int ops = batch * orows * ocols * odepth;
    long int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_id_x < ops) {
        // output pixel at batch=ob, coord=(or,oc) at map=oz
        int orcd=orows*ocols*odepth; // out size of batch
        int orc=orows*ocols;  // out size of slice
        int ob=thread_id_x/orcd; // current batch (ib=ob)
        int bm=thread_id_x%orcd; // index in batch
        int ouz=bm/orc; // out depth (iuz=ouz)
        int our=(bm%orc)/ocols; // out row
        int ouc=(bm%orc)%ocols; // out col

        int inr = our * sr;  // in row
        int inc = ouc * sc;  // in col
        int ircd=irows*icols*idepth; // in size of batch
        int irc=irows*icols;  // in size of batch

        int min_i = -padrt;
        int max_i = irows+padrb-kr;
        int i = min_i + inr;  // row

        int min_j = -padcl;
        int max_j = icols+padcr-kc;
        int j = min_j + inc;  // column

        int b = ob;  // batch
        int k = ouz;  // depth
        int p = thread_id_x;  // index

        // Check bounds
        if (i <= max_i && j <= max_j){

            int px = indX[p];
            int py = indY[p];
            int pz = k;

            if (px < 0){}
            else if (py < 0){}
            else if (px >= icols){}
            else if (py >= irows){}
            else {
                // Compute address from indices (row-major)
                int ptr = (b * ircd) + (pz * irc) + (py * icols) + px;
                atomicAdd(&ID[ptr], D[p]);
            }

        }
    }

}

// AvgPool
__global__ void avgpool2d(float* I, int batch, int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt, int padrb,int padcl, int padcr) {
    long int ops = batch * orows * ocols * odepth;
    long int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int ksize = kr*kc;

    if (thread_id_x < ops) {
        // Parse "thread_id_x" to output(b, d, r, c) ****************
        // output pixel at batch=ob, coord=(or,oc) at map=oz
        int orcd=orows*ocols*odepth; // out size of batch
        int orc=orows*ocols;  // out size of slice
        int ob=thread_id_x/orcd; // batch's index => B[i] || (ib=ob)
        int bm=thread_id_x%orcd; // index inside batch i => thread_id_x=23, batch_size=20: index = 3
        int ouz=bm/orc; // depth index (iuz=ouz)
        int our=(bm%orc)/ocols; // row index
        int ouc=(bm%orc)%ocols; // col index

        // Parse output(b, d, r, c) to input(b, d, r, c) ****************
        int inr = our * sr;  // input row index (without padding)
        int inc = ouc * sc;  // input col index (without padding)
        int ircd=irows*icols*idepth; // in size of batch
        int irc=irows*icols;  // in size of batch

        int min_i = -padrt;
        int max_i = irows+padrb-kr;
        int i = min_i + inr;  // input row index (with padding)

        int min_j = -padcl;
        int max_j = icols+padcr-kc;
        int j = min_j + inc;  // input column index (with padding)

        int b = ob;  // batch
        int k = ouz;  // depth
        int p = thread_id_x;  // index

        // Check bounds
        if (i <= max_i && j <= max_j){

            // Sum values window
            float sum = 0.0f;
            for (int ki = 0; ki < kr; ki++){ // rows (kernel): top-bottom
                for (int kj = 0; kj < kc; kj++) { // cols (kernel): left-right

                    // Get value W[ki,kj] value in window
                    int px = j + kj;
                    int py = i + ki;
                    int pz = k;
                    float v = 0.0;

                    if (px < 0) v = 0.0;
                    else if (py < 0) v = 0.0;
                    else if (px >= icols) v = 0.0;
                    else if (py >= irows) v = 0.0;
                    else {
                        int ptr = (b * ircd) + (pz * irc) + (py * icols) + px;
                        v = I[ptr];
                    }

                    sum+=v;
                } // kernel cols
            } // kernel rows

            // Set output value
            O[p] = sum/(float)ksize;
        }
    }
}

// AvgPool backward
__global__ void avgpool2d_back(float* D, float* ID, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt, int padrb,int padcl, int padcr) {

    long int ops = batch * orows * ocols * odepth;
    long int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int ksize = kr*kc;

    if (thread_id_x < ops) {
        // output pixel at batch=ob, coord=(or,oc) at map=oz
        int orcd=orows*ocols*odepth; // out size of batch
        int orc=orows*ocols;  // out size of slice
        int ob=thread_id_x/orcd; // current batch (ib=ob)
        int bm=thread_id_x%orcd; // index in batch
        int ouz=bm/orc; // out depth (iuz=ouz)
        int our=(bm%orc)/ocols; // out row
        int ouc=(bm%orc)%ocols; // out col

        int inr = our * sr;  // in row
        int inc = ouc * sc;  // in col
        int ircd=irows*icols*idepth; // in size of batch
        int irc=irows*icols;  // in size of batch

        int min_i = -padrt;
        int max_i = irows+padrb-kr;
        int i = min_i + inr;  // row

        int min_j = -padcl;
        int max_j = icols+padcr-kc;
        int j = min_j + inc;  // column

        int b = ob;  // batch
        int k = ouz;  // depth
        int p = thread_id_x;  // index

        // Check bounds
        if (i <= max_i && j <= max_j){

            // Add values (delta equally distributed among all the values in the window)
            for (int ki = 0; ki < kr; ki++){  // top-bottom (kernel)
                for (int kj = 0; kj < kc; kj++) {  // left-right (kernel)

                    int px = j+kj;
                    int py = i+ki;
                    int pz = k;

                    if (px < 0){}
                    else if (py < 0){}
                    else if (px >= icols){}
                    else if (py >= irows){}
                    else {
                        // Compute address from indices (row-major)
                        int ptr = (b * ircd) + (pz * irc) + (py * icols) + px;
                        atomicAdd(&ID[ptr], D[p]/(float)ksize);
                    }

                }
            }

        }
    }

}