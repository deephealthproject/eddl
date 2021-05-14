/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>
#include <limits>       // std::numeric_limits

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"


float get_pixel(int b,int px,int py,int pz,PoolDescriptor *D,int isize,int irsize) {
  // Check boundaries of the window
  if (px<0) return MIN_FLOAT;
  if (py<0) return MIN_FLOAT;
  if (px>=D->ic) return MIN_FLOAT;
  if (py>=D->ir) return MIN_FLOAT;

  // Compute address from indices (row-major)
  unsigned int address = (b*isize) + (pz*irsize) + (py*D->ic) + px;
  return D->I->ptr[address];
}

void add_pixel(int b,int px,int py,int pz,PoolDescriptor *D,int isize,int irsize,float val) {
  // Check boundaries of the window
  if (px<0) return;
  if (py<0) return;
  if (px>=D->ic) return;
  if (py>=D->ir) return;

  // Compute address from indices (row-major)
  unsigned int address = (b*isize) + (pz*irsize) + (py*D->ic) + px;
  D->ID->ptr[address]+=val;
}

float get_pixel3d(int in, int iz, int id, int ir, int ic, PoolDescriptor3D *D, int stride_b, int stride_d, int stride_r, int stride_c) {
    // Check boundaries of the window
    if (id<0) return MIN_FLOAT;
    if (ir<0) return MIN_FLOAT;
    if (ic<0) return MIN_FLOAT;

    if (id>=D->id) return MIN_FLOAT;
    if (ir>=D->ir) return MIN_FLOAT;
    if (ic>=D->ic) return MIN_FLOAT;

    // Compute address from indices (row-major)
    unsigned int address = (in* stride_b) + (iz* stride_d) + (id* stride_r) + (ir* stride_c) + ic;
    return D->I->ptr[address];
}

void add_pixel3d(int in, int iz, int id, int ir, int ic, PoolDescriptor3D *D, int stride_b, int stride_d, int stride_r, int stride_c, float val) {
    // Check boundaries of the window
    if (id<0) return;
    if (ir<0) return;
    if (ic<0) return;

    if (id>=D->id) return;
    if (ir>=D->ir) return;
    if (ic>=D->ic) return;

    // Compute address from indices (row-major)
    unsigned int address = (in* stride_b) + (iz* stride_d) + (id* stride_r) + (ir* stride_c) + ic;
    D->ID->ptr[address]+=val;
}

void cpu_mpool2D(PoolDescriptor *D){
    _profile(_CPU_MPOOL2D, 0);
    int isize = D->ir*D->ic*D->iz;
    int irsize = D->ir*D->ic;

    #pragma omp parallel for default(none) shared(D, isize, irsize)
    for(int b=0; b<D->I->shape[0]; b++){  // Batches
        int p=b*D->size;  // Kernel's index (opt. shared variable)

        for(int k=0; k<D->iz; k++) { // Depth: front-back
            for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
                for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc, p++) { // cols: left-right

                    // Get max value in window
                    float max = CPU_LOWEST_FLOAT;
                    for(int ki=0; ki<D->kr; ki++){  // rows (kernel): top-bottom
                        for(int kj=0; kj<D->kc; kj++) { // cols (kernel): left-right

                            // Get value W[ki,kj] value in window
                            float v = get_pixel(b,j+kj,i+ki, k, D, isize, irsize);
                            if (v>max) {
                                max = v;
                                D->indX->ptr[p] = j+kj;
                                D->indY->ptr[p] = i+ki;
                            }

                        } // kernel cols
                    }  // kernel rows

                    // Set output value
                    D->O->ptr[p] = max;

                } // cols
            } // rows
        } // depth
    } // batch
    _profile(_CPU_MPOOL2D, 1);
}

void cpu_mpool2D_back(PoolDescriptor *D){
    _profile(_CPU_MPOOL2D_BACK, 0);
    int isize = D->ir*D->ic*D->iz;
    int irsize = D->ir*D->ic;

    #pragma omp parallel for default(none) shared(D, isize, irsize)
    for(int b=0; b<D->I->shape[0]; b++){  // Batches (ob=ib)
        int p=b*D->size; // Kernel's index (opt. shared variable)

        for(int k=0; k<D->iz; k++) { // Channels
            for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
                for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc, p++) { // cols: left-right

                    int x = D->indX->ptr[p];  // previous: j+kj
                    int y = D->indY->ptr[p];  // previous: i+ki
                    add_pixel(b, x, y, k, D, isize, irsize, D->D->ptr[p]);  // Set input's delta

                } // cols
            } // rows
        } // depth
    } // batch
    _profile(_CPU_MPOOL2D_BACK, 1);
}

void cpu_mpool3D(PoolDescriptor3D *D){
//    _profile(_CPU_MPOOL2D, 0);
    int stride_b = D->iz*D->id*D->ir*D->ic;
    int stride_d = D->id*D->ir*D->ic;
    int stride_r = D->ir*D->ic;
    int stride_c = D->ic;


    #pragma omp parallel for default(none) shared(D, stride_b, stride_d, stride_r, stride_c)
    for(int b=0; b<D->I->shape[0]; b++){  // Batches
        int p=b*D->size;  // Kernel's index (opt. shared variable)

        for(int c=0; c<D->iz; c++) { // Channels

            for(int d=-D->paddf; d<=D->id+D->paddb-D->kd; d+=D->sd) {  // Depth: front-back
                for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
                    for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc, p++) { // cols: left-right

                        // Get max value in window
                        float max = CPU_LOWEST_FLOAT;
                        for(int kd=0; kd<D->kd; kd++){  // depth (kernel): front-back
                            for(int ki=0; ki<D->kr; ki++){  // rows (kernel): top-bottom
                                for(int kj=0; kj<D->kc; kj++) { // cols (kernel): left-right

                                    // Get value W[ki,kj] value in window
                                    float v = get_pixel3d(b, c, d+kd, i+ki, j+kj, D, stride_b, stride_d, stride_r, stride_c);
                                    if (v>max) {
                                        max = v;
                                        D->indZ->ptr[p] = d+kd;
                                        D->indY->ptr[p] = i+ki;
                                        D->indX->ptr[p] = j+kj;
                                    }

                                } // kernel cols
                            }  // kernel rows
                        }  // kernel depth

                        // Set output value
                        D->O->ptr[p] = max;

                    } // cols
                } // rows
            } // depth
        } // channels
    } // batch
//    _profile(_CPU_MPOOL3D, 1);
}

void cpu_mpool3D_back(PoolDescriptor3D *D){
//    _profile(_CPU_MPOOL3D_BACK, 0);
    int stride_b = D->iz*D->id*D->ir*D->ic;
    int stride_d = D->id*D->ir*D->ic;
    int stride_r = D->ir*D->ic;
    int stride_c = D->ic;

#pragma omp parallel for default(none) shared(D, stride_b, stride_d, stride_r, stride_c)
    for(int b=0; b<D->I->shape[0]; b++){  // Batches
        int p=b*D->size;  // Kernel's index (opt. shared variable)

        for(int c=0; c<D->iz; c++) { // Channels
            for(int d=-D->paddf; d<=D->id+D->paddb-D->kd; d+=D->sd) {  // Depth: front-back
                for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
                    for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc, p++) { // cols: left-right

                        int z = D->indZ->ptr[p];  // previous: d+kd
                        int y = D->indY->ptr[p];  // previous: i+ki
                        int x = D->indX->ptr[p];  // previous: j+kj
                        add_pixel3d(b, c, z, y, x, D, stride_b, stride_d, stride_r, stride_c, D->D->ptr[p]);  // Set input's delta

                    } // cols
                } // rows
            } // depth
        } // channels
    } // batch

//    _profile(_CPU_MPOOL3D_BACK, 1);
}


void cpu_avgpool2D(PoolDescriptor *D){
    _profile(_CPU_AVGPOOL2D, 0);
    int isize = D->ir*D->ic*D->iz;
    int irsize = D->ir*D->ic;
    int ksize = D->kr*D->kc;

    #pragma omp parallel for default(none) shared(D, isize, irsize, ksize)
    for(int b=0; b<D->I->shape[0]; b++){  // Batches
        int p=b*D->size; // Kernel's index (opt. shared variable)

        for(int k=0; k<D->iz; k++) { // Depth: front-back
            for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
                for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc, p++) { // cols: left-right

                    // Sum values window
                    float sum = 0.0f;
//                    int ksize_local = 0;
                    for(int ki=0; ki<D->kr; ki++){  // rows (kernel): top-bottom
                        for(int kj=0; kj<D->kc; kj++) { // cols (kernel): left-right

                            // Check boundaries of the window
                            int px = j+kj;
                            int py = i+ki;
                            if (px>=0 && py>=0 && px<D->ic && py<D->ir){
                                // Get value W[ki,kj] value in window
                                sum += get_pixel(b,px,py, k, D, isize, irsize);
//                                ksize_local += 1;
                            }


                        } // kernel cols
                    }  // kernel rows

                    // Set output value
                    D->O->ptr[p]= sum/(float)ksize;//_local;

                } // cols
            } // rows
        } // depth
    } // batch
    _profile(_CPU_AVGPOOL2D, 1);
}

void cpu_avgpool2D_back(PoolDescriptor *D){
    _profile(_CPU_AVGPOOL2D_BACK, 0);
    int isize = D->ir*D->ic*D->iz;
    int irsize = D->ir*D->ic;
    int ksize = D->kr*D->kc;

    #pragma omp parallel for default(none) shared(D, isize, irsize, ksize)
    for(int b=0; b<D->I->shape[0]; b++){  // Batches
        int p=b*D->size; // Kernel's index (opt. shared variable)

        for(int k=0; k<D->iz; k++) { // Depth: front-back
            for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
                for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc,p++) { // cols: left-right

                    // Walk kernel window to equally distribute the delta among all the elements
                    for(int ki=0; ki<D->kr; ki++){  // rows (kernel): top-bottom
                        for(int kj=0; kj<D->kc; kj++) { // cols (kernel): left-right

                            float v = D->D->ptr[p]/(float)ksize;  // This should be ksize_local, but storing/computing it would increase the cost
                            add_pixel(b, j + kj, i + ki, k, D, isize, irsize, v);
                        } // kernel cols
                    }  // kernel rows

                } // cols
            } // rows
        } // depth
    } // batch
    _profile(_CPU_AVGPOOL2D_BACK, 1);
}

void cpu_avgpool3D(PoolDescriptor3D *D){
//    _profile(_CPU_MPOOL2D, 0);
    int stride_b = D->iz*D->id*D->ir*D->ic;
    int stride_d = D->id*D->ir*D->ic;
    int stride_r = D->ir*D->ic;
    int stride_c = D->ic;
    float ksize = (float)D->kd*D->kr*D->kc;


#pragma omp parallel for default(none) shared(D, stride_b, stride_d, stride_r, stride_c, ksize)
    for(int b=0; b<D->I->shape[0]; b++){  // Batches
        int p=b*D->size;  // Kernel's index (opt. shared variable)

        for(int c=0; c<D->iz; c++) { // Channels

            for(int d=-D->paddf; d<=D->id+D->paddb-D->kd; d+=D->sd) {  // Depth: front-back
                for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
                    for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc, p++) { // cols: left-right

                        // Sum values window
                        float sum = 0.0f;
//                        int ksize_local = 0;
                        for(int kd=0; kd<D->kd; kd++){  // depth (kernel): front-back
                            for(int ki=0; ki<D->kr; ki++){  // rows (kernel): top-bottom
                                for(int kj=0; kj<D->kc; kj++) { // cols (kernel): left-right

                                    // Check boundaries of the window
                                    int id = d+kd;
                                    int ir = i+ki;
                                    int ic = j+kj;
                                    if (id>=0 && ir>=0 && ic>=0 && id<D->id && ir<D->ir && ir<D->ic){
                                        // Get value W[ki,kj] value in window
                                        sum += get_pixel3d(b, c, id, ir, ic, D, stride_b, stride_d, stride_r, stride_c);
//                                        ksize_local += 1;
                                    }

                                } // kernel cols
                            }  // kernel rows
                        }  // kernel depth

                        // Set output value
                        D->O->ptr[p] = sum/(float)ksize;//_local;

                    } // cols
                } // rows
            } // depth
        } // channels
    } // batch
//    _profile(_CPU_MPOOL3D, 1);
}

void cpu_avgpool3D_back(PoolDescriptor3D *D){
//    _profile(_CPU_MPOOL3D_BACK, 0);
    int stride_b = D->iz*D->id*D->ir*D->ic;
    int stride_d = D->id*D->ir*D->ic;
    int stride_r = D->ir*D->ic;
    int stride_c = D->ic;
    float ksize = (float)D->kd*D->kr*D->kc;

#pragma omp parallel for default(none) shared(D, stride_b, stride_d, stride_r, stride_c, ksize)
    for(int b=0; b<D->I->shape[0]; b++){  // Batches
        int p=b*D->size;  // Kernel's index (opt. shared variable)

        for(int c=0; c<D->iz; c++) { // Channels
            for(int d=-D->paddf; d<=D->id+D->paddb-D->kd; d+=D->sd) {  // Depth: front-back
                for(int i=-D->padrt; i<=D->ir+D->padrb-D->kr; i+=D->sr) {  // rows: top-bottom
                    for(int j=-D->padcl; j<=D->ic+D->padcr-D->kc; j+=D->sc, p++) { // cols: left-right

                        // Walk kernel window to equally distribute the delta among all the elements
                        for(int kd=0; kd<D->kd; kd++){  // depth (kernel): front-back
                            for(int ki=0; ki<D->kr; ki++){  // rows (kernel): top-bottom
                                for(int kj=0; kj<D->kc; kj++) { // cols (kernel): left-right

                                    float v = D->D->ptr[p]/(float)ksize;  // This should be ksize_local, but storing/computing it would increase the cost
                                    add_pixel3d(b, c, d+kd, i+ki, j+kj, D, stride_b, stride_d, stride_r, stride_c, v);  // Set input's delta
                                } // kernel cols
                            }  // kernel rows
                        }  // kernel depth

                    } // cols
                } // rows
            } // depth
        } // channels
    } // batch

//    _profile(_CPU_MPOOL3D_BACK, 1);
}