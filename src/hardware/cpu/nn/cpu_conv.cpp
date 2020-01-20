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

void im2col(int b,ConvolDescriptor *D,float *ptrI,int col2im)
{
  int i,j,k;
  int pz,py,px,y,x;
  int ksize=D->kr*D->kc;
  int kr2=D->kr/2;
  int kc2=D->kc/2;

  if (kc2==0) kc2=-1;

  int orsize=D->r*D->c;

  int isize=D->ir*D->ic*D->iz;
  int irsize=D->ir*D->ic;

  k=0;
  py=-D->padrt;
  px=-D->padcl;


  for(j=0;j<D->matI.rows();j++) {
    k=j;

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
    if (px>=D->ic+D->padcl-kc2-1) {
      px=-D->padcl;
      py+=D->sr;
    }
  }
}


void cpu_conv2D(ConvolDescriptor *D)
{
  int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  float *ptrO=D->O->ptr;
  float *ptrI=D->ptrI;

  // Map memory to Eigen
  new(&D->matK) Eigen::Map<Eigen::MatrixXf>(D->K->ptr, D->kr * D->kc * D->kz, D->nk);

  #pragma omp parallel for
  for(int b=0;b<D->I->shape[0];b++){

    float *ptrO=D->O->ptr+(b*osize);
    float *ptrI=D->ptrI+(b*isize);

    Eigen::Map<Eigen::MatrixXf> matI=Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);
    Eigen::Map<Eigen::MatrixXf> matO=Eigen::Map<Eigen::MatrixXf>(ptrO,D->r*D->c,D->z);

    im2col(b,D,ptrI,0);

    matO=matI*D->matK;
  }// batch

  //bias
  #pragma omp parallel for
  for(int b=0;b<D->O->shape[0];b++) {
    float *ptrO=D->O->ptr+(b*osize);
    for(int z=0;z<D->O->shape[1];z++)
      for(int r=0;r<D->O->shape[2];r++)
        for(int c=0;c<D->O->shape[3];c++,ptrO++)
            (*ptrO)+=D->bias->ptr[z];
  }

}

void cpu_conv2D_grad(ConvolDescriptor *D)
{
  //return;
  int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  // Map memory to Eigen
  new(&D->matgK) Eigen::Map<Eigen::MatrixXf>(D->gK->ptr, D->kr * D->kc * D->kz, D->nk);

  #pragma omp parallel for
  for(int b=0;b<D->I->shape[0];b++){

    float *ptrD=D->D->ptr+(b*osize);
    float *ptrI=D->ptrI+(b*isize);

    Eigen::Map<Eigen::MatrixXf> matI=Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);
    Eigen::Map<Eigen::MatrixXf> matD=Eigen::Map<Eigen::MatrixXf>(ptrD,D->r*D->c,D->z);

    D->matgK+=matI.transpose()*matD;
  }// batch

  //bias

  #pragma omp parallel for 
  for(int b=0;b<D->D->shape[0];b++) {
    float *ptrD=D->D->ptr+(b*osize);
    for(int z=0;z<D->D->shape[1];z++)
      for(int r=0;r<D->D->shape[2];r++)
        for(int c=0;c<D->D->shape[3];c++,ptrD++)
            D->gbias->ptr[z]+=(*ptrD);

    }

}

void cpu_conv2D_back(ConvolDescriptor *D)
{
  int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  float *ptrD=D->D->ptr;
  float *ptrI=D->ptrI;

  // Map memory to Eigen
  new(&D->matK) Eigen::Map<Eigen::MatrixXf>(D->K->ptr, D->kr * D->kc * D->kz, D->nk);
  new (&(D->matI)) Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);

  #pragma omp parallel for
  for(int b=0;b<D->I->shape[0];b++){

    float *ptrD=D->D->ptr+(b*osize);
    float *ptrI=D->ptrI+(b*isize);

    Eigen::Map<Eigen::MatrixXf> matI=Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);
    Eigen::Map<Eigen::MatrixXf> matD=Eigen::Map<Eigen::MatrixXf>(ptrD,D->r*D->c,D->z);

    matI=matD*D->matK.transpose();

    im2col(b,D,ptrI,1);

  }// batch
}
