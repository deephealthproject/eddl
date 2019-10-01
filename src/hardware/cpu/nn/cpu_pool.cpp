
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


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
      for(i=-D->padr;i<=D->ir+D->padr-D->kr;i+=D->sr) {
        for(j=-D->padc;j<=D->ic+D->padc-D->kc;j+=D->sc,p++) {
           float max=0;
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
      for(i=-D->padr;i<=D->ir+D->padr-D->kr;i+=D->sr) {
        for(j=-D->padc;j<=D->ic+D->padc-D->kc;j+=D->sc,p++) {
           int x=D->indX->ptr[p];
           int y=D->indY->ptr[p];
           add_pixel(b,x,y,k,D,isize,irsize,D->D->ptr[p]);
        }
      }
    } // depth
  }// batch

}

