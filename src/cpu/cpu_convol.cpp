#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "cpu_convol.h"

float get_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize)
{
  if (px<0.0) return 0.0;
  if (py<0.0) return 0.0;
  if (px>=D->ic) return 0.0;
  if (py>=D->ir) return 0.0;

  int p=(b*isize)+(pz*irsize)+(py*D->ic)+px;
  return D->I->ptr[p];
}

void add_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize,float val)
{
  if (px<0.0) return;
  if (py<0.0) return;
  if (px>=D->ic) return;
  if (py>=D->ir) return;

  int p=(b*isize)+(pz*irsize)+(py*D->ic)+px;
  D->ID->ptr[p]+=val;
}

void im2col(int b,ConvolDescriptor *D,int col2im)
{
  int i,j,k;
  int pz,py,px,y,x;
  int ksize=D->kr*D->kc;
  int kr2=D->kr/2;
  int kc2=D->kc/2;


  int orsize=D->r*D->c;

  int isize=D->ir*D->ic*D->iz;
  int irsize=D->ir*D->ic;

  k=0;
  py=-D->padr;
  px=-D->padc;


  float *ptrI=&(D->matI(0,0));
  for(j=0;j<D->matI.rows();j++) {
    k=j;
    //fprintf(stderr,"%d %d\n",py,px);
    //getchar();
    for(i=0;i<D->matI.cols();i++,k+=orsize) {
       pz=i/ksize;
       y=py+(i%ksize)/D->kc;
       x=px+(i%D->kc);

       if(col2im)
         add_pixel(b,x,y,pz,D,isize,irsize,ptrI[k]);
       else
         ptrI[k]=get_pixel(b,x,y,pz,D,isize,irsize);

      }
    px+=D->sc;
    if (px>=D->ic+D->padc-kc2-1) {
      px=-D->padc;
      py+=D->sr;
    }
  }
}

void cpu_conv2D(ConvolDescriptor *D)
{
  int osize=D->z*D->r*D->c;
  float *ptrO=D->O->ptr;
  for(int b=0;b<D->I->sizes[0];b++,ptrO+=osize){
    im2col(b,D,0);

    new (&(D->matO)) Eigen::Map<Eigen::MatrixXf>(ptrO,D->r*D->c,D->z);

    D->matO=D->matI*D->matK;
  }// batch
}

void cpu_conv2D_grad(ConvolDescriptor *D)
{
  int osize=D->z*D->r*D->c;
  float *ptrD=D->D->ptr;
  for(int b=0;b<D->I->sizes[0];b++,ptrD+=osize){
    im2col(b,D,0);

    new (&(D->matD)) Eigen::Map<Eigen::MatrixXf>(ptrD,D->r*D->c,D->z);

    D->matgK+=D->matI.transpose()*D->matD;
  }// batch
}

void cpu_conv2D_back(ConvolDescriptor *D)
{
  int osize=D->z*D->r*D->c;
  float *ptrD=D->D->ptr;
  for(int b=0;b<D->I->sizes[0];b++,ptrD+=osize){
    new (&(D->matD)) Eigen::Map<Eigen::MatrixXf>(ptrD,D->r*D->c,D->z);
    D->matI=D->matD*D->matK.transpose();

    im2col(b,D,1);

  }// batch
}



















//////////////////////////////////
