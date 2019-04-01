#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "cpu_convol.h"

float get_pixel(int b,int px,int py,int pz,Tensor *A,ConvolDescriptor *D)
{
  if (px<0.0) return 0.0;
  if (py<0.0) return 0.0;
  if (px>=D->ic) return 0.0;
  if (py>=D->ir) return 0.0;

  int p=b*(D->ir*D->ic*D->iz)+pz*(D->ir*D->ic)+py*(D->ic)+px;
  return A->ptr[p];
}


void cpu_conv2D(Tensor *A,ConvolDescriptor *D,Tensor *C)
{
  int i,j,k,n,m;

  int pz,py,px,y,x;
  int ksize=(D->kr*D->kc);
  int osize=(D->z*D->r*D->c);

  for(int b=0;b<A->sizes[0];b++){
    float *ptr=C->ptr;
    ptr+=b*osize;
    new (&(D->matC)) Eigen::Map<Eigen::MatrixXf>(ptr,D->r*D->c,D->z);

    //  matA=Eigen::MatrixXf(r*c,kr*kc*kz);
    k=0;
    py=-D->padr;
    px=-D->padc;
    for(j=0;j<D->matA.rows();j++) {
      for(i=0;i<D->matA.cols();i++,k++) {
         pz=i/ksize;
         y=py+(i%ksize)/D->kc;
         x=px+(i%D->kc);
         D->ptr[k]=get_pixel(b,x,y,pz,A,D);
         px++;
        }
      px++;
      if (px>=D->ic+D->padc) {
        px=-D->padc;
        py++;
      }
    }

    fprintf(stderr,"%dx%d\n",D->matC.rows(),D->matC.cols());
    fprintf(stderr,"%dx%d\n",D->matA.rows(),D->matA.cols());
    fprintf(stderr,"%dx%d\n",D->matK.rows(),D->matK.cols());

    D->matC=D->matA*D->matK;
  }// batch
}




















//////////////////////////////////
