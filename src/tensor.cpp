// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019 Roberto Paredes Palacios, <rparedes@dsic.upv.es>

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
#include <stdlib.h>
#include <math.h>
#include <initializer_list>
#include <vector>
#include <string>
#include <iostream>
#include "tensor.h"

using namespace std;

void msg(string s,string s2)
{
  cout<<"\n"<<s<<s2<<"\n";
  exit(0);
}
void msg(string s){msg(s,"");}


// Tensor class
Tensor::Tensor():device(DEV_CPU),dim(0),tam(0){}

Tensor::Tensor(const initializer_list<int>& init):Tensor(init,DEV_CPU){}
Tensor::Tensor(const initializer_list<int>& init, string t):Tensor(shape(init.begin(), init.end()),DEV_CPU,t){}
Tensor::Tensor(const initializer_list<int>& init, int dev):Tensor(shape(init.begin(), init.end()),dev,std::string("FLOAT32")){}
Tensor::Tensor(const initializer_list<int>& init, int dev,string t):Tensor(shape(init.begin(), init.end()),dev,t){}


Tensor::Tensor(const shape s):Tensor(s,DEV_CPU){}
Tensor::Tensor(const shape s,string t):Tensor(s,DEV_CPU,t){}
Tensor::Tensor(const shape s,int dev):Tensor(s,dev,std::string("FLOAT32")){}

Tensor::Tensor(shape s,int dev,string t)
{
  #ifndef cGPU
  if (dev==DEV_GPU){
    fprintf(stderr,"Not compiled for GPU\n");
    exit(0);
  }
  #endif
  #ifndef cFPGA
  if (dev==DEV_FPGA){
    fprintf(stderr,"Not compiled for FPGA\n");
    exit(0);
  }
  #endif

  device=dev;
  dim=s.size();
  sizes=s;
  if (t=="FLOAT32") type=FLOAT32;
  else if (t=="FLOAT64") type=FLOAT64;
  else if (t=="INT32") type=INT32;
  else msg("Tensor unkown type",t);

  tam=1;
  for(int i=0;i<dim;++i) tam*=s[i];

  if (device==DEV_CPU) {
    if (dim<3) mem(type);
    else {
      ptr=(Tensor **)malloc(sizes[0]*sizeof(Tensor *));
      s.erase(s.begin());
      for(int i=0;i<sizes[0];++i)
        ptr[i]=new Tensor(s,device);
    }
  }
  #ifdef cGPU
  else if (device==DEV_GPU) mem(type);
  #endif
}

///////////////////////////////////////////
void Tensor::clean(int t)
{
  if (device==DEV_CPU) {
    if (dim==1) {
      if (t==FLOAT32) ptr1f.resize(0);
      if (t==FLOAT64) ptr1d.resize(0);
      if (t==INT32) ptr1i.resize(0);
    }
    else if (dim==2) {
      if (t==FLOAT32) ptr2f.resize(0,0);
      if (t==FLOAT64) ptr2d.resize(0,0);
      if (t==INT32) ptr2i.resize(0,0);
    }
  }
  #ifdef cGPU
  else if (device==DEV_GPU) {
    if (t==FLOAT32) delete_tensor(gptrd);
    if (t==FLOAT64) delete_tensor(gptrf);
    if (t==INT32) delete_tensor(gptri);
  }
  #endif
}

void Tensor::mem(int t)
{
  if (device==DEV_CPU) {
  if (dim==1) {
    if (t==FLOAT32) ptr1f.resize(sizes[0]);
    if (t==FLOAT64) ptr1d.resize(sizes[0]);
    if (t==INT32) ptr1i.resize(sizes[0]);
  }
  else if (dim==2) {
    if (t==FLOAT32) ptr2f.resize(sizes[0],sizes[1]);
    if (t==FLOAT64) ptr2d.resize(sizes[0],sizes[1]);
    if (t==INT32) ptr2i.resize(sizes[0],sizes[1]);
  }
}
  #ifdef cGPU
  else if (device==DEV_GPU) mem(type);
  #endif
}

///////////////////////////////////////////
void Tensor::changetype(int t)
{
  if (type==t) return;

  if (device==DEV_CPU) {
    if (dim==1) {clean(type);mem(t);}
    else if (dim==2) {clean(type);mem(t);}
    else
      for(int i=0;i<sizes[0];++i)
        ptr[i]->changetype(t);
  }
  #ifdef cGPU
  else if (device==DEV_GPU) {clean(type);mem(t);}
  #endif
  type=t;
}

///////////////////////////////////////////
Tensor *Tensor::clone()
{
  Tensor *C=new Tensor(getshape(),device);
  if (type!=FLOAT32)
    C->changetype(type);
  return C;

}

///////////////////////////////////////////
Tensor::~Tensor()
{
  if (device==DEV_CPU) {
    if (dim<3) clean(type);
    else {
      for(int i=0;i<sizes[0];++i)
        delete ptr[i];
      delete ptr;
    }
  }
  #ifdef cGPU
  else if (device==DEV_GPU) clean(type);
  #endif
}

///////////////////////////////////////////

shape Tensor::getshape()
{
  shape s=sizes;
  return s;
}

///////////////////////////////////////////
void Tensor::rand(){
  if (device==DEV_CPU) {
    if (dim==1) {
      if (type==FLOAT32) for(int i=0;i<sizes[0];++i) ptr1f(i)=(float)(std::rand()%1000)/(float)1000.0;
      if (type==FLOAT64) for(int i=0;i<sizes[0];++i) ptr1d(i)=(double)(std::rand()%1000)/(double)1000.0;
      if (type==INT32) for(int i=0;i<sizes[0];++i) ptr1i(i)=std::rand()%1000;
    }
    else if (dim==2) {
      if (type==FLOAT32) for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2f(i,j)=(float)(std::rand()%1000)/(float)1000.0;
      if (type==FLOAT64) for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2d(i,j)=(double)(std::rand()%1000)/(double)1000.0;
      if (type==INT32) for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2i(i,j)=std::rand()%1000;
    }
    else
      for(int i=0;i<sizes[0];++i)
        ptr[i]->rand();

  }
}

///////////////////////////////////////////
void Tensor::info()
{
  int i;

  fprintf(stderr,"DIM=%d\n",dim);
  fprintf(stderr,"(");
  for (i = 0; i < dim-1; i++)
		fprintf(stderr,"%d,",sizes[i]);
  fprintf(stderr,"%d)\n",sizes[i]);

  if (type==FLOAT32) fprintf(stderr,"Type FLOAT32\nTotal bytes=%ld\n",tam*sizeof(float));
  if (type==FLOAT64) fprintf(stderr,"Type FLOAT64\nTotal bytes=%ld\n",tam*sizeof(double));
  if (type==INT32) fprintf(stderr,"Type INT32\nTotal bytes=%ld\n",tam*sizeof(int));

  if (device==DEV_CPU) fprintf(stderr,"Device=CPU\n");
  else if (device==DEV_GPU) fprintf(stderr,"Device=GPU\n");
  else fprintf(stderr,"Device=FPGA\n");
}


void Tensor::print(){

  if (device==DEV_CPU) {
    if (dim==1) {
      if (type==FLOAT32) cout<<ptr1f;
      if (type==FLOAT64) cout<<ptr1d;
      if (type==INT32) cout<<ptr1i;
      cout<<"\n";
    }
    else if (dim==2) {
      if (type==FLOAT32) cout<<ptr2f;
      if (type==FLOAT64) cout<<ptr2d;
      if (type==INT32) cout<<ptr2i;
      cout<<"\n";
    }
    else
      for(int i=0;i<sizes[0];++i) {
        ptr[i]->print();
        cout<<"\n";
      }
  }
}











///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////































//////
