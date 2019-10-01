
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
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  float *ptrO=D->O->ptr;
  float *ptrI=D->ptrI;

  for(int b=0;b<D->I->shape[0];b++,ptrO+=osize,ptrI+=isize){
    new (&(D->matI)) Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);
    new (&(D->matO)) Eigen::Map<Eigen::MatrixXf>(ptrO,D->r*D->c,D->z);
    im2col(b,D,0);
    D->matO=D->matI*D->matK;
  }// batch

  //bias

  ptrO=D->O->ptr;
  for(int b=0;b<D->O->shape[0];b++)
    for(int z=0;z<D->O->shape[1];z++)
      for(int r=0;r<D->O->shape[2];r++)
        for(int c=0;c<D->O->shape[3];c++,ptrO++)
            (*ptrO)+=D->bias->ptr[z];

}

void cpu_conv2D_grad(ConvolDescriptor *D)
{
  //return;
  int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  float *ptrD=D->D->ptr;
  float *ptrI=D->ptrI;

  for(int b=0;b<D->I->shape[0];b++,ptrD+=osize,ptrI+=isize){
    // re-using previous lowering
    new (&(D->matI)) Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);
    new (&(D->matD)) Eigen::Map<Eigen::MatrixXf>(ptrD,D->r*D->c,D->z);

    D->matgK+=D->matI.transpose()*D->matD;
  }// batch

  //bias
  ptrD=D->D->ptr;
  for(int b=0;b<D->D->shape[0];b++)
    for(int z=0;z<D->D->shape[1];z++)
      for(int r=0;r<D->D->shape[2];r++)
        for(int c=0;c<D->D->shape[3];c++,ptrD++)
            D->gbias->ptr[z]+=(*ptrD);

}

void cpu_conv2D_back(ConvolDescriptor *D)
{
  int osize=D->z*D->r*D->c;
  int isize=D->r*D->c*D->kc*D->kr*D->kz;//r*c,kr*kc*kz

  float *ptrD=D->D->ptr;
  float *ptrI=D->ptrI;
  new (&(D->matI)) Eigen::Map<Eigen::MatrixXf>(ptrI,D->r*D->c,D->kz*D->kr*D->kc);

  for(int b=0;b<D->I->shape[0];b++,ptrD+=osize){
    new (&(D->matD)) Eigen::Map<Eigen::MatrixXf>(ptrD,D->r*D->c,D->z);

    D->matI=D->matD*D->matK.transpose();

    im2col(b,D,1);

  }// batch
}


