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

#include "gpu_nn_kernels.h"
#include "../gpu_kernels.h"



__global__ void maxpool2d(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padr, int padc, float* indX, float* indY) {

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

        int min_i = -padr;
        int max_i = irows+padr-kr;
        int i = min_i + inr;  // row

        int min_j = -padc;
        int max_j = icols+padc-kc;
        int j = min_j + inc;  // column

        int b = ob;  // batch
        int k = ouz;  // depth
        int p = thread_id_x;  // index

        // Check bounds
        if (i <= max_i && j <= max_j){

            // Get maximum value in the kernel window
            float max = 0;
            for (int ki = 0; ki < kr; ki++)  // kernel_rows
                for (int kj = 0; kj < kc; kj++) {  // kernel_cols

                    // Get pixel
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

                    if (v > max) {
                        max = v;
                        indX[p] = j + kj;
                        indY[p] = i + ki;
                    }
                }
            O[p] = max;
        }
    }

}

__global__ void maxpool2d_back(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, int sr,int sc,int padr, int padc, float* indX, float* indY, float* D, float* ID){

    int isize=irows * icols * idepth;
    int irsize=irows * icols;

    long int ops = batch * isize;
    long int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_id_x < ops) {

        int b=thread_id_x/isize; // current batch (ib=ob)
        int bm=thread_id_x%isize; // index in batch
        int z=bm/irsize; // out depth (iuz=ouz)
        int r=(bm%irsize)/icols; // out row
        int c=(bm%irsize)%icols; // out col

        int inr = r * sr;  // in row
        int inc = c * sc;  // in col

        int min_i = -padr;
        int max_i = irows+padr-kr;
        int i = min_i + inr;  // row

        int min_j = -padc;
        int max_j = icols+padc-kc;
        int j = min_j + inc;  // column

        int p = thread_id_x;  // index

        // Check bounds
        if (i <= max_i && j <= max_j){
            int px=indX[p];
            int py=indY[p];
            int pz=z;


            if (px>=0.0 && py>=0.0 && px<icols && p<irows){
                int p=(b*isize)+(pz*irsize)+(py*icols)+px;
                ID[p]+=D[p]; // +val
            }

        }
    }

}
