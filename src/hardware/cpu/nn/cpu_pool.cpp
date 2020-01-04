/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "cpu_nn.h"


void cpu_mpool2D(PoolDescriptor *D)
{
  int i,j,k,ki,kj;
  int isize=D->ir*D->ic*D->iz;
  int irsize=D->ir*D->ic;

  int p=0;
  for(int b=0;b<D->I->shape[0];b++){
    for(k=0;k<D->iz;k++) {
      for(i=-D->padrt;i<=D->ir+D->padrb-D->kr;i+=D->sr) {
        for(j=-D->padcl;j<=D->ic+D->padcr-D->kc;j+=D->sc,p++) {
           float max=get_pixel(b,j,i,k,D,isize,irsize);
           D->indX->ptr[p]=j;
           D->indY->ptr[p]=i;
           for(ki=0;ki<D->kr;ki++)
             for(kj=0;kj<D->kc;kj++) {
               float v=get_pixel(b,j+kj,i+ki,k,D,isize,irsize);
               if (v>max) {
                 max=v;
                 D->indX->ptr[p]=j+kj;
                 D->indY->ptr[p]=i+ki;
               }
              }
           D->O->ptr[p]=max;
        }
      }
    } // depth
  }// batch
}

void cpu_mpool2D_back(PoolDescriptor *D)
{
  int i,j,k,ki,kj;
  int isize=D->ir*D->ic*D->iz;
  int irsize=D->ir*D->ic;

  int p=0;
  for(int b=0;b<D->I->shape[0];b++){
    for(k=0;k<D->iz;k++) {
      for(i=-D->padrt;i<=D->ir+D->padrb-D->kr;i+=D->sr) {
        for(j=-D->padcl;j<=D->ic+D->padcr-D->kc;j+=D->sc,p++) {
           int x=D->indX->ptr[p];
           int y=D->indY->ptr[p];
           add_pixel(b,x,y,k,D,isize,irsize,D->D->ptr[p]);
        }
      }
    } // depth
  }// batch

}
