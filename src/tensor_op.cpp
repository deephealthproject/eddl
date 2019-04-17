// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <initializer_list>
#include <vector>
#include <string>
#include <iostream>
#include "tensor.h"

#include "cpu/cpu_convol.h"

#ifdef cGPU
#include "gpu/tensor_cuda.h"
#include "gpu/tensor_cuda_op.h"
#endif

using namespace std;

///////////////////////////////////////////
/// TENSOR OPERATIONS AS STATIC METHODS ///
///////////////////////////////////////////

int Tensor::eqsize(Tensor *A, Tensor *B)
{
  if (A->dim!=B->dim) return 0;

  for(int i=0;i<A->dim;i++)
    if (A->sizes[i]!=B->sizes[i]) return 0;

  return 1;
}


///////////////////////////////////////
/// Copy from A to B
//////////////////////////////////////
void Tensor::copy(Tensor *A, Tensor *B)
{

  if (!Tensor::eqsize(A,B))
    msg("Tensors with different sizes","Tensor::copy");

  B->tsem->lock();
  if ((A->isCPU())&&(B->isCPU()))
    {
      for(int i=0;i<A->tam;i++)
        B->ptr[i]=A->ptr[i];

    }
  #ifdef cGPU
  else if ((A->isGPU())&&(B->isGPU())) {
    gpu_copy_gpu(A,B);
  }
  else if ((A->isCPU())&&(B->isGPU()))
    {
      gpu_copy_to_gpu(A->ptr,B);
    }
  else if ((A->isGPU())&&(B->isCPU()))
    {
      gpu_copy_from_gpu(A,B->ptr);
    }
  #endif
  else
    {
      fprintf(stderr,"(%d %d)\n",A->device,B->device);
      msg("unsupported copy between devices","Tensor::copy");
    }
  B->tsem->unlock();
}

///////////////////////////////////////
/// Partial copy dim=1
//////////////////////////////////////
void Tensor::fill(Tensor *A, int aini,int aend,Tensor *B,int bini,int bend,int inc)
{
  if (A->dim!=B->dim)
    msg("Tensors with different sizes","Tensor::fill");

  B->tsem->lock();
  if ((A->isCPU())&&(B->isCPU()))
    {
      int at=A->tam/A->sizes[0];
      int bt=B->tam/B->sizes[0];

      int t=1;
      for(int i=2;i<A->dim;i++)
        t*=A->sizes[i];

      for(int i=0;i<A->sizes[0];i++) {
        int ap=(i*at)+(aini*t);
        int bp=(i*bt)+(bini*t);

        for(int j=aini;j<aend;j++) {
          for(int k=0;k<t;k++,ap++,bp++)
            if (inc) B->ptr[bp]+=A->ptr[ap];
            else B->ptr[bp]=A->ptr[ap];
        }
      }
    }
  #ifdef cGPU
  else if ((A->isGPU())&&(B->isGPU())) {
    gpu_fill(A,aini,aend,B,bini,bend,inc);
  }
  #endif
  else
    {
      fprintf(stderr,"(%d %d)\n",A->device,B->device);
      msg("unsupported copy between devices","Tensor::copy");
    }
  B->tsem->unlock();
}


void Tensor::inc(Tensor *A, Tensor *B)
{

  if (!Tensor::eqsize(A,B))
    msg("Tensors with different sizes","Tensor::inc");


  if ((A->isCPU())&&(B->isCPU()))
    {
      B->tsem->lock();

      for(int i=0;i<A->tam;i++)
        B->ptr[i]+=A->ptr[i];

      B->tsem->unlock();
    }
  #ifdef cGPU
  else if ((A->isGPU())&&(B->isGPU())) {
    Tensor::sum(1,A,1,B,B,0);
  }
  else if (((A->isCPU())&&(B->isGPU()))||((A->isGPU())&&(B->isCPU())))
    {
       Tensor *n=new Tensor(B->getshape(),B->device);
       Tensor::copy(A,n);
       Tensor::sum(1,n,1,B,B,0);
       delete n;
    }
  #endif
  else
    {
      fprintf(stderr,"(%d %d)\n",A->device,B->device);
      msg("unsupported copy between devices","Tensor::inc");
    }
}


///////////////////////////////////////
/// Select from A to B
//////////////////////////////////////
void Tensor::select(Tensor *A, Tensor *B,vector<int> sind,int ini,int end)
{

  if ((A->tam/A->sizes[0])!=(B->tam/B->sizes[0])) msg("Incompatible sizes","Tensor::select");

  //B->tsem->lock();
  if ((A->isCPU())&&(B->isCPU()))
    {
      int s=A->tam/A->sizes[0];

      for(int i=ini;i<end;i++) {
        int p=sind[i]*s;
        int pb=(i-ini)*s;
        for(int j=0;j<s;j++,p++,pb++)
          B->ptr[pb]=A->ptr[p];
      }
  }
  else {
    msg("unsuppoted select between devices","Tensor::select");
  }
  //B->tsem->unlock();
}


///////////////////////////////////////
//// MULT2D C=A*B
//// tA means transpose A {0,1}
//// tB means transpose B {0,1}
//// tC 1 means C+=A*B (increment over C)
//// Dimensions and types must be compatible
//// Only for 2D Tensors
///////////////////////////////////////
void Tensor::mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC)
{

  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::mult2D");
  if ((A->dim!=2)||(B->dim!=2)||(C->dim!=2)) msg("Only 2D tensors","Tensor::mult2D");
  if (!tA)
    {
      if (!tB)
        {
          if ((A->sizes[1]!=B->sizes[0])||(A->sizes[0]!=C->sizes[0])||(B->sizes[1]!=C->sizes[1])) msg("Incompatible dims","Tensor::mult2D");
        }
      else
        if ((A->sizes[1]!=B->sizes[1])||(A->sizes[0]!=C->sizes[0])||(B->sizes[0]!=C->sizes[1])) msg("Incompatible dims","Tensor::mult2D");
    }
  else
    {
      if (!tB)
        {
          if ((A->sizes[0]!=B->sizes[0])||(A->sizes[1]!=C->sizes[0])||(B->sizes[1]!=C->sizes[1])) msg("Incompatible dims","Tensor::mult2D");
        }
      else
        if ((A->sizes[0]!=B->sizes[1])||(A->sizes[1]!=C->sizes[0])||(B->sizes[0]!=C->sizes[1])) msg("Incompatible dims","Tensor::mult2D");
    }


  C->tsem->lock();

  if (A->isCPU())
    {

      if (!tB)
        {
          if (!tA)
            {
              if (!incC) *(C->ptr2)=*(B->ptr2)*(*(A->ptr2));
              else *(C->ptr2)+=*(B->ptr2)*(*(A->ptr2));
            }
          else
            {
              if (!incC) *(C->ptr2)=*(B->ptr2)*((*(A->ptr2)).transpose());
              else *(C->ptr2)+=*(B->ptr2)*((*(A->ptr2)).transpose());
            }
        }
      else
        {
          if (!tA)
            {
              if (!incC) *(C->ptr2)=(*(B->ptr2)).transpose()*(*(A->ptr2));
              else *(C->ptr2)+=(*(B->ptr2)).transpose()*(*(A->ptr2));
            }
          else
            {
              if (!incC) *(C->ptr2)=(*(B->ptr2)).transpose()*((*(A->ptr2)).transpose());
              else *(C->ptr2)+=(*(B->ptr2)).transpose()*((*(A->ptr2)).transpose());
            }
        }
    }

#ifdef cGPU
  else if (A->isGPU())
    {
      gpu_mult2D(A,tA,B,tB,C,incC);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  C->tsem->unlock();
}


///////////////////////////////////////
//// Element Mult C=A.*B
//// incC 1 means C+=A.*B (increment over C)
//// Dimensions must be compatible
///////////////////////////////////////
void Tensor::el_mult(Tensor *A, Tensor *B, Tensor *C,int incC)
{
  C->tsem->lock();
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::el_mult");
  if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims","Tensor::el_mult");

  if (A->isCPU())
    {
        for(int i=0;i<A->tam;i++)
         if (incC) C->ptr[i]+=A->ptr[i]*B->ptr[i];
         else C->ptr[i]=A->ptr[i]*B->ptr[i];
    }
#ifdef cGPU
  else if (A->isGPU())
    {
       gpu_el_mult(A,B,C,incC);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  C->tsem->unlock();
}


///////////////////////////////////////
//// Element Div C=A./B
//// incC 1 means C+=A./B (increment over C)
//// Dimensions must be compatible
///////////////////////////////////////
void Tensor::el_div(Tensor *A, Tensor *B, Tensor *C,int incC)
{

  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::el_div");
  if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims","Tensor::el_div");

  C->tsem->lock();
  if (A->isCPU())
    {

      for(int i=0;i<A->tam;i++)
        if (incC) C->ptr[i]+=A->ptr[i]/B->ptr[i];
        else C->ptr[i]=A->ptr[i]/B->ptr[i];
    }
#ifdef cGPU
  else if (A->isGPU())
    {
      gpu_el_div(A,B,C,incC);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  C->tsem->unlock();
}


///////////////////////////////////////
//// sum C=(sca*A)+(scb*B)
//// or C+=(sca*A)+(scb*B) if incC is 1
//// Dimensions and types must be compatible
///////////////////////////////////////
void Tensor::sum(float scA, Tensor *A, float scB, Tensor *B, Tensor *C,int incC)
{
  int aux=0;

  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::sum");
  if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims","Tensor::sum");

  C->tsem->lock();
  if (A->isCPU())
  {

    for(int i=0;i<A->tam;i++)
      if (incC) C->ptr[i]+=scA*A->ptr[i]+scB*B->ptr[i];
      else C->ptr[i]=scA*A->ptr[i]+scB*B->ptr[i];
  }
#ifdef cGPU
  else if (A->isGPU())
    {
      gpu_sum(scA,A,scB,B,C,incC);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  C->tsem->unlock();
}

void Tensor::sum(Tensor *A,Tensor *B, Tensor *C)
{
  Tensor::sum(1.0,A,1.0,B,C,0);
}



///////////////////////////////////////
//// sum2D_rowise C=A.rowise+B
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
///////////////////////////////////////
void Tensor::sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C)
{
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::sum2D_rowwise");
  if ((A->dim!=2)||(B->dim!=1)||(C->dim!=2)) msg("sum2D_rowwise dims");
  if ((!eqsize(A,C))||(A->sizes[1]!=B->sizes[0])) msg("Incompatible dims","Tensor::sum2D_rowwise");

  C->tsem->lock();
  if (A->isCPU())
    {
      int p=0;
      for(int i=0;i<A->sizes[0];i++) {
        for(int j=0;j<A->sizes[1];j++,p++)
          C->ptr[p]=A->ptr[p]+B->ptr[j];
        }
    }
#ifdef cGPU
  else if (A->isGPU())
    {
      gpu_sum2D_rowwise(A,B,C);

    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  C->tsem->unlock();
}


///////////////////////////////////////
//// sum2D_colwise C=A.colwise+B
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
///////////////////////////////////////
void Tensor::sum2D_colwise(Tensor *A, Tensor *B, Tensor *C)
{
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::sum2D_colwise");
  if ((A->dim!=2)||(B->dim!=1)||(C->dim!=2)) msg("sum2D_colwise dims");
  if ((!eqsize(A,C))||(A->sizes[0]!=B->sizes[0])) msg("Incompatible dims","Tensor::sum2D_colwise");

  C->tsem->lock();
  if (A->isCPU())
  {
    int p=0;
    for(int i=0;i<A->sizes[0];i++){
      for(int j=0;j<A->sizes[1];j++,p++)
        C->ptr[p]=A->ptr[p]+B->ptr[i];
      }
  }
#ifdef cGPU
  else if (A->isGPU())
    {
      gpu_sum2D_colwise(A,B,C);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  C->tsem->unlock();
}


///////////////////////////////////////
//// reduce_sum2D B=reduce_sum2D(A)
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
//// axis is the dimension to be sumed
///////////////////////////////////////
void Tensor::reduce_sum2D(Tensor *A, Tensor *B, int axis,int incB)
{
  if (A->device!=B->device) msg("Tensors in different devices","Tensor::reduce_sum2D");
  if ((A->dim-1)!=B->dim) msg("Incorrect dims","Tensor::reduce_sum2D");
  if ((A->sizes[1-axis]!=B->sizes[0])) msg("Incompatible dims","Tensor::reduce_sum2D");

  B->tsem->lock();
  if (A->isCPU())
    {
      if (axis==0)
      {
        if (!incB) for(int i=0;i<A->sizes[1];++i) B->ptr[i]=0;

        int p=0;
        for(int i=0;i<A->sizes[0];++i) {
          for(int j=0;j<A->sizes[1];++j,p++)
               B->ptr[j]+=A->ptr[p];
          }

      }
      else
        {
          if (!incB) for(int i=0;i<A->sizes[0];++i) B->ptr[i]=0;

          int p=0;
          for(int i=0;i<A->sizes[0];++i) {
            for(int j=0;j<A->sizes[1];++j,p++)
                 B->ptr[i]+=A->ptr[p];
            }
        }
    }
#ifdef cGPU
  else if (A->isGPU())
    {
      gpu_reduce_sum2D(A,B,axis,incB);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  B->tsem->unlock();
}


////////////////////////////////
//// CONVOLUTIONS

ConvolDescriptor::ConvolDescriptor(const initializer_list<int>& ks,const initializer_list<int>& st,const initializer_list<int>& p )
{
  ksize=vector<int>(ks.begin(), ks.end());
  stride=vector<int>(st.begin(), st.end());
  pad=vector<int>(p.begin(), p.end());

  if (ksize.size()!=3) msg("Kernels must have 3 dimensions","ConvolDescriptor::ConvolDescriptor");
  if (stride.size()!=2) msg("Strides must have 2 dimensions","ConvolDescriptor::ConvolDescriptor");
  if (pad.size()!=2) msg("Padding must have 2 dimensions","ConvolDescriptor::ConvolDescriptor");
}


ConvolDescriptor::ConvolDescriptor(const vector<int>& ks, const vector<int>& st, string p)
{
    if (ks.size()!=3){ msg("Kernels must have 3 dimensions","ConvolDescriptor::ConvolDescriptor");}
    if (st.size()!=2){ msg("Strides must have 2 dimensions","ConvolDescriptor::ConvolDescriptor");}

    if (p=="same") {
        pad.push_back(ks[1]/2);
        pad.push_back(ks[2]/2);
    }
    else if (p=="none") {
        pad.push_back(0);
        pad.push_back(0);
    }
    else msg("Incorrect padding type","ConvolDescriptor::ConvolDescriptor");

}


ConvolDescriptor::ConvolDescriptor(const initializer_list<int>& ks,const initializer_list<int>& st, string p)
{
    ksize=vector<int>(ks.begin(), ks.end());
    stride=vector<int>(st.begin(), st.end());
    ConvolDescriptor(ksize, stride, p);
}


void ConvolDescriptor::build(Tensor *A)
{
  if (A->dim!=4) msg("Tensors are not 4D","ConvolDescriptor::build");

  I=A;

  nk=ksize[0];
  kr=ksize[1];
  kc=ksize[2];
  kz=A->sizes[1];

  sr=stride[0];
  sc=stride[1];

  iz=A->sizes[1];
  ir=A->sizes[2];
  ic=A->sizes[3];

  padr=pad[0];
  padc=pad[1];

  z=nk;
  r=(ir-kr+2*padr)/sr+1;
  c=(ic-kc+2*padc)/sc+1;

  if ((r<=0)||(c<=0))
    msg("Invalid output sizes","ConvolDescriptor::build");

  O=new Tensor({A->sizes[0],z,r,c},A->device);
  D=new Tensor({A->sizes[0],z,r,c},A->device);

}

void ConvolDescriptor::params()
{
  K=new Tensor({nk,kz,kr,kc},I->device);
  bias=new Tensor({nk},I->device);
  gK=new Tensor({nk,kz,kr,kc},I->device);
  gbias=new Tensor({nk},I->device);

  if (I->isCPU()) {
    // mem for ptr, lowering im2col
    //eigen mat in col-major
    matI=Eigen::MatrixXf(r*c,kr*kc*kz);
    new (&matK) Eigen::Map<Eigen::MatrixXf>(K->ptr,kr*kc*kz,nk);
    new (&matgK) Eigen::Map<Eigen::MatrixXf>(gK->ptr,kr*kc*kz,nk);
    // convolution: matC=matA*matK
  }

}


////////////////////////////////
////  POOLING

PoolDescriptor::PoolDescriptor(const initializer_list<int>& ks,const initializer_list<int>& st,const initializer_list<int>& p )
{
  ksize=vector<int>(ks.begin(), ks.end());
  stride=vector<int>(st.begin(), st.end());
  pad=vector<int>(p.begin(), p.end());

  if (ksize.size()!=2) msg("Pooling Kernels must have 2 dimensions","PoolDescriptor::PoolDescriptor");
  if (stride.size()!=2) msg("Strides must have 2 dimensions","PoolDescriptor::PoolDescriptor");
  if (pad.size()!=2) msg("Padding must have 2 dimensions","PoolDescriptor::PoolDescriptor");
}

PoolDescriptor::PoolDescriptor(const initializer_list<int>& ks,const initializer_list<int>& st, string p)
{
  ksize=vector<int>(ks.begin(), ks.end());
  stride=vector<int>(st.begin(), st.end());

  if (ksize.size()!=2) msg("Pooling Kernels must have 2 dimensions","PoolDescriptor::PoolDescriptor");
  if (stride.size()!=2) msg("Strides must have 2 dimensions","PoolDescriptor::PoolDescriptor");

  if (p=="same") {
    pad.push_back(ksize[1]/2);
    pad.push_back(ksize[2]/2);
  }
  else if (p=="none") {
      pad.push_back(0);
      pad.push_back(0);
  }
  else msg("Incorrect padding type","PoolDescriptor::PoolDescriptor");
}


void PoolDescriptor::build(Tensor *A)
{
  if (A->dim!=4) msg("Tensors are not 4D","PoolDescriptor::build");

  I=A;


  kr=ksize[0];
  kc=ksize[1];

  sr=stride[0];
  sc=stride[1];

  iz=A->sizes[1];
  ir=A->sizes[2];
  ic=A->sizes[3];

  padr=pad[0];
  padc=pad[1];

  z=iz;
  r=(ir-kr+2*padr)/sr+1;
  c=(ic-kc+2*padc)/sc+1;

  if ((r<=0)||(c<=0))
    msg("Invalid output sizes","PoolDescriptor::build");

  O=new Tensor({A->sizes[0],z,r,c},A->device);
  D=new Tensor({A->sizes[0],z,r,c},A->device);

}

/////////////////////////////////////////////////////////////////////
//// Conv2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D(ConvolDescriptor *D)
{
  if ((D->I->dim!=4)) msg("Tensors are not 4D","Tensor::Conv2D");

  D->O->tsem->lock();
  if (D->I->isCPU())
    {
       cpu_conv2D(D);
    }
#ifdef cGPU
  else if (D->I->isGPU())
    {
       //gpu_conv2D(A,B,D,C);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  D->O->tsem->unlock();
}
/////////////////////////////////////////////////////////////////////
//// Conv2D Grad
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D_grad(ConvolDescriptor *D)
{
  if ((D->I->dim!=4)) msg("Tensors are not 4D","Tensor::Conv2D");

  D->gK->tsem->lock();
  if (D->I->isCPU())
    {
       D->gK->set(0.0);
       cpu_conv2D_grad(D);
    }
#ifdef cGPU
  else if (D->I->isGPU())
    {
       //gpu_conv2D(A,B,D,C);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  D->gK->tsem->unlock();
}
/////////////////////////////////////////////////////////////////////
//// Conv2D Back
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D_back(ConvolDescriptor *D)
{
  if ((D->I->dim!=4)) msg("Tensors are not 4D","Tensor::Conv2D");

  D->ID->tsem->lock();
  if (D->I->isCPU())
    {
       cpu_conv2D_back(D);
    }
#ifdef cGPU
  else if (D->I->isGPU())
    {
       //gpu_conv2D(A,B,D,C);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  D->ID->tsem->unlock();
}


/////////////////////////////////////////////////////////////////////
//// MPool2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::MPool2D(PoolDescriptor *D)
{
  if ((D->I->dim!=4)) msg("Tensors are not 4D","Tensor::MPool2D");

  D->O->tsem->lock();
  if (D->I->isCPU())
    {
       cpu_mpool2D(D);
    }
#ifdef cGPU
  else if (D->I->isGPU())
    {
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  D->O->tsem->unlock();
}

/////////////////////////////////////////////////////////////////////
//// MPool2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::MPool2D_back(PoolDescriptor *D)
{
  if ((D->I->dim!=4)) msg("Tensors are not 4D","Tensor::MPool2D_back");

  D->ID->tsem->lock();
  if (D->I->isCPU())
    {
       cpu_mpool2D_back(D);
    }
#ifdef cGPU
  else if (D->I->isGPU())
    {
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  D->ID->tsem->unlock();
}


////////////////////////////////
/// COST FUNCTIONS
////////////////////////////////
// Cross-Entropy: C=-(A*log(B)+(1-A)*log(1-B))
void Tensor::cent(Tensor *A,Tensor *B, Tensor *C)
{
  if (A->device!=B->device) msg("Tensors in different devices","Tensor::cross-entropy");
  if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims","Tensor::cross-entropy");

  C->tsem->lock();
  if (A->isCPU())
    {

      for(int i=0;i<A->tam;i++) {
        C->ptr[i]=0;
        if (A->ptr[i]!=0.0) C->ptr[i]-=A->ptr[i]*log(B->ptr[i]);
        if (A->ptr[i]!=1.0) C->ptr[i]-=(1.0-A->ptr[i])*log(1.0-B->ptr[i]);
      }
  }
#ifdef cGPU
  else if (A->isGPU())
    {
       gpu_cent(A,B,C);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  C->tsem->unlock();
}


////////////////////////////////
/// METRICS FUNCTIONS
////////////////////////////////
int Tensor::accuracy(Tensor *A,Tensor *B)
{
  if (A->device!=B->device) msg("Tensors in different devices","Tensor::accuracy");
  if (!eqsize(A,B)) msg("Incompatible dims","Tensor::accuracy");
  if (A->dim!=2)  msg("Accuracy only over 2D Tensor (batch x probs)","Tensor::Accuracy");

  int acc=0;

  B->tsem->lock();

  if (A->isCPU())
    {
      int aind,bind;

      for(int i=0;i<A->sizes[0];i++)
        {
          (*A->ptr2).col(i).maxCoeff(&aind);
          (*B->ptr2).col(i).maxCoeff(&bind);
          if (aind==bind) acc++;
       }
    }
#ifdef cGPU
  else if (A->isGPU())
    {
       gpu_accuracy(A,B,&acc);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  B->tsem->unlock();
  return acc;

}


////////////////////////////////
/// ACTIVATIONS
////////////////////////////////
// RELU
void Tensor::ReLu(Tensor *A,Tensor *B)
{
  if (A->device!=B->device) msg("Tensors in different devices","Tensor::ReLu");
  if (!eqsize(A,B)) msg("Incompatible dims","Tensor::ReLu");

  B->tsem->lock();
  if (A->isCPU())
    {

      for(int i=0;i<A->tam;i++) {
        if (A->ptr[i]>0.0) B->ptr[i]=A->ptr[i];
        else B->ptr[i]=0.0;
      }
    }
#ifdef cGPU
  else if (A->isGPU())
    {
    gpu_relu(A,B);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif

  B->tsem->unlock();
}


// RELU Derivative, always increment over parent delta
void Tensor::D_ReLu(Tensor *D, Tensor *I, Tensor *PD)
{
  if ((D->device!=I->device)||(D->device!=PD->device)) msg("Tensors in different devices","Tensor::D_ReLu");
  if ((!eqsize(D,I))||(!eqsize(D,PD))) msg("Incompatible dims","Tensor::D_ReLu");

  PD->tsem->lock();
  if (D->isCPU())
  {

    for(int i=0;i<D->tam;i++) {
      if (I->ptr[i]>0.0) PD->ptr[i]=D->ptr[i];
      else PD->ptr[i]=0.0;
    }
  }
#ifdef cGPU
  else if (D->isGPU())
    {
      gpu_d_relu(D,I,PD);

    }
#endif
#ifdef cFPGA
  else {

  }
#endif
  PD->tsem->unlock();
}


// SOFTMAX
void Tensor::Softmax(Tensor *A,Tensor *B)
{
  if (A->device!=B->device) msg("Tensors in different devices","Tensor::Softmax");
  if (!eqsize(A,B)) msg("Incompatible dims","Tensor::Softmax");
  if (A->dim!=2)  msg("Softmax only over 2D Tensor (batch x logits)","Tensor::Softmax");

  B->tsem->lock();

  if (A->isCPU())
    {
      float max,sum;



      for(int i=0;i<A->sizes[0];i++)
        {

          max=(*A->ptr2).col(i).maxCoeff();
          for(int j=0;j<A->sizes[1];j++)
            (*B->ptr2)(j,i)=exp((*A->ptr2)(j,i)-max);

          sum=(*B->ptr2).col(i).sum();
          for(int j=0;j<B->sizes[1];j++)
            (*B->ptr2)(j,i)=(*B->ptr2)(j,i)/sum;
        }
    }
#ifdef cGPU
  else if (A->isGPU())
    {
      gpu_softmax(A,B);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif

  B->tsem->unlock();
}


// SOFTMAX DERIVATIVE
void Tensor::D_Softmax(Tensor *D,Tensor *I,Tensor *PD)
{
  if ((D->device!=I->device)||(D->device!=PD->device)) msg("Tensors in different devices","Tensor::D_Softmax");
  if ((!eqsize(D,I))||(!eqsize(D,PD))) msg("Incompatible dims","Tensor::D_Softmax");
  if (D->dim!=2) msg("D_Softmax only over 2D Tensor (batch x delta_probs)","Tensor::D_Softmax");


  if (D->isCPU())
    {
      PD->tsem->lock();

      for(int i=0;i<D->tam;i++)
         PD->ptr[i]+=D->ptr[i]*(I->ptr[i]*(1.0-I->ptr[i]));

      PD->tsem->unlock();
    }
#ifdef cGPU
  else if (D->isGPU())
    {

      Tensor *aux=new Tensor(D->getshape(),D->device);
      aux->set(1.0);
      Tensor::sum(1.0,aux,-1.0,I,aux,0);
      Tensor::el_mult(I,aux,aux,0);
      Tensor::el_mult(D,aux,PD,1);

      delete aux;
    }
#endif
#ifdef cFPGA
  else {

  }
#endif



}


///////////////////////////

//////
