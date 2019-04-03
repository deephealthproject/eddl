#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "cpu_convol.h"

float get_pixel(int b,int px,int py,int pz,Tensor *A,ConvolDescriptor *D,int isize,int irsize)
{
  if (px<0.0) return 0.0;
  if (py<0.0) return 0.0;
  if (px>=D->ic) return 0.0;
  if (py>=D->ir) return 0.0;

  int p=(b*isize)+(pz*irsize)+(py*D->ic)+px;
  return A->ptr[p];
}


void cpu_conv2D(Tensor *A,ConvolDescriptor *D)
{
  int i,j,k;

  int pz,py,px,y,x;
  int ksize=D->kr*D->kc;
  int kr2=D->kr/2;
  int kc2=D->kc/2;

  int osize=D->z*D->r*D->c;
  int orsize=D->r*D->c;

  int isize=D->ir*D->ic*D->iz;
  int irsize=D->ir*D->ic;

  float *ptrI=&(D->matI(0,0));
  float *ptrO=D->O->ptr;
  for(int b=0;b<A->sizes[0];b++,ptrO+=osize){

    new (&(D->matO)) Eigen::Map<Eigen::MatrixXf>(ptrO,D->r*D->c,D->z);

    k=0;
    py=-D->padr;
    px=-D->padc;
    for(j=0;j<D->matI.rows();j++) {
      k=j;
      //fprintf(stderr,"%d %d\n",py,px);
      //getchar();
      for(i=0;i<D->matI.cols();i++,k+=orsize) {
         pz=i/ksize;
         y=py+(i%ksize)/D->kc;
         x=px+(i%D->kc);

         ptrI[k]=get_pixel(b,x,y,pz,A,D,isize,irsize);
        }
      px+=D->sc;
      if (px>=D->ic+D->padc-kc2-1) {
        px=-D->padc;
        py+=D->sr;
      }
    }

    D->matO=D->matI*D->matK;
  }// batch
}


















//////////////////////////////////
