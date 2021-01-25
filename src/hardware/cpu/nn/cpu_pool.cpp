/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
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
  if (px<0) return 0.0;
  if (py<0) return 0.0;
  if (px>=D->ic) return 0.0;
  if (py>=D->ir) return 0.0;

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

        for(int k=0; k<D->iz; k++) { // Depth: front-back (oz=iz)
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
                    for(int ki=0; ki<D->kr; ki++){  // rows (kernel): top-bottom
                        for(int kj=0; kj<D->kc; kj++) { // cols (kernel): left-right

                            // Get value W[ki,kj] value in window
                            float v = get_pixel(b,j+kj,i+ki, k, D, isize, irsize);
                            sum += v;

                        } // kernel cols
                    }  // kernel rows

                    // Set output value
                    D->O->ptr[p]= sum/(float)ksize;

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
                            add_pixel(b, j + kj, i + ki, k, D, isize, irsize, D->D->ptr[p]/(float)ksize);
                        } // kernel cols
                    }  // kernel rows

                } // cols
            } // rows
        } // depth
    } // batch
    _profile(_CPU_AVGPOOL2D_BACK, 1);
}
