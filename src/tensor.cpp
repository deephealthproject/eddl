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
#include "utils.h"

#ifdef cGPU
#include "gpu/tensor_cuda.h"
#include "gpu/tensor_cuda_op.h"
#endif

using namespace std;

int initcuda[MAX_GPUS]={0,0,0,0,0,0,0,0};
int linpos;

extern ostream& operator<<(ostream& os, const shape s);

void msg(string s,string s2)
{
  cout<<"\n"<<s<<" ("<<s2<<")\n";
  exit(0);
}


void msg(string s){msg(s,"");}

// Tensor class
Tensor::Tensor():device(DEV_CPU),dim(0),tam(0){}

Tensor::Tensor(const initializer_list<int>& init):Tensor(init,DEV_CPU){}
Tensor::Tensor(const initializer_list<int>& init, int dev):Tensor(shape(init.begin(), init.end()),dev){}

Tensor::Tensor(const shape s):Tensor(s,DEV_CPU){}
Tensor::Tensor(shape s,int dev)
{
#ifndef cGPU
  if ((dev>DEV_CPU)&&(device<DEV_FPGA))
    {
      fprintf(stderr,"Not compiled for GPU\n");
      exit(0);
    }
#endif
#ifndef cFPGA
  if (dev>=DEV_FPGA)
    {
      fprintf(stderr,"Not compiled for FPGA\n");
      exit(0);
    }
#endif

  device=dev;
  dim=s.size();
  sizes=s;

  tam=1;
  for(int i=0;i<dim;++i) tam*=s[i];

  if (device==DEV_CPU)
    {
      if (dim==1) ptr1.resize(sizes[0]);
      else if (dim==2) ptr2.resize(sizes[0],sizes[1]);
      else
        {
          ptr=(Tensor **)malloc(sizes[0]*sizeof(Tensor *));
          s.erase(s.begin());
          for(int i=0;i<sizes[0];++i)
            ptr[i]=new Tensor(s,device);
        }
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      gpu_device=device-DEV_GPU;
      if (!initcuda[gpu_device])
        {

          gpu_init(gpu_device);
          initcuda[gpu_device]=1;
        }
      //gpu_set_device(gpu_device);
      gptr=gpu_create_tensor(gpu_device,tam);
    }
#endif
#ifdef cFPGA
  else {
    // create FPGA Tensor
  }
#endif

  tsem=new mutex();
}


///////////////////////////////////////////
void Tensor::load(FILE *fe)
{
  int i,j;

  if (dim==1)
    {
      float *fptr=(float *)malloc(sizes[0]*sizeof(float ));
      int read=fread(fptr,sizeof(float),sizes[0],fe);
      if (read!=(sizes[0]))
        {
          fprintf(stderr,"Error reading file (%d!=%d)\nCheck format\n",read,sizes[1]);
          exit(1);
        }
      for(j=0;j<sizes[0];j++)
        ptr1(j)=fptr[j];
      free(fptr);
    }
  else if (dim==2)
    {
      float *fptr=(float *)malloc(sizes[1]*sizeof(float ));
      for(i=0;i<sizes[0];i++)
        {
          if (feof(fe)) {fprintf(stderr,"Error reading line %d\n",i);exit(1);}
          int read=fread(fptr,sizeof(float),sizes[1],fe);
          if (read!=(sizes[1]))
            {
              fprintf(stderr,"Error reading file (%d!=%d)\nCheck format\n",read,sizes[1]);
              exit(1);
            }
          for(j=0;j<sizes[1];j++)
            ptr2(i,j)=fptr[j];

        }
      free(fptr);
    }
  else
    for(i=0;i<sizes[0];i++)
      ptr[i]->load(fe);

}


/////////////////////////////////////////////////////////////////////////
Tensor::Tensor(string fname)
{
  FILE *fe;
  int i,j,v;
  float *fptr;

  fe=fopen(fname.c_str(),"rb");
  if (fe==NULL)
    {
      fprintf(stderr,"%s not found\n",fname.c_str());
      exit(1);
    }

  int read=fread(&dim,sizeof(int),1,fe);
  for(int i=0;i<dim;++i)
    {
      int read=fread(&v,sizeof(int),1,fe);
      sizes.push_back(v);
    }
  shape s=sizes;

  cout<<"loading file with tensor:"<<s<<"\n";

  device=DEV_CPU;
  tam=1;
  tsem=new mutex();
  for(int i=0;i<dim;++i) tam*=sizes[i];
  if (dim==1) ptr1.resize(sizes[0]);
  else if (dim==2) ptr2.resize(sizes[0],sizes[1]);
  else
    {
      ptr=(Tensor **)malloc(sizes[0]*sizeof(Tensor *));
      s.erase(s.begin());
      for(int i=0;i<sizes[0];++i)
        ptr[i]=new Tensor(s,device);
    }

  load(fe);
  fclose(fe);
}


///////////////////////////////////////////
void Tensor::save(FILE *fe)
{
  int i,j;
  float fv;

  if (dim==1)
    {
      for(i=0;i<sizes[0];i++)
        {
          fv=ptr1(i);
          fwrite(&fv, sizeof(float),1,fe);
        }
    }
  else if (dim==2)
    {
      for(i=0;i<sizes[0];i++)
        for(j=0;j<sizes[1];j++)
          {
            fv=ptr2(i,j);
            fwrite(&fv, sizeof(float),1,fe);
          }
    }
  else
    for(int i=0;i<sizes[0];i++)
      ptr[i]->save(fe);
}


///////////////////////////////////////////
void Tensor::save(string fname)
{
  if (device!=DEV_CPU)
    msg("Only save CPU Tensors","Tensor::save");

  int i,j;
  FILE *fe;
  float fv;

  fe=fopen(fname.c_str(),"wb");
  if (fe==NULL)
    {
      fprintf(stderr,"Not abel to write %s \n",fname.c_str());
      exit(1);
    }

  fprintf(stderr,"writting bin file\n");

  fwrite(&dim, sizeof(int),1,fe);
  for(i=0;i<dim;++i)
    fwrite(&sizes[i], sizeof(int),1,fe);

  save(fe);

  fclose(fe);

}


///////////////////////////////////////////
Tensor *Tensor::share()
{
  Tensor *C=new Tensor(getshape(),device);

  return C;

}


///////////////////////////////////////////
Tensor::~Tensor()
{
  if (device==DEV_CPU)
    {
      if (dim==1) ptr1.resize(0);
      else if (dim==2) ptr2.resize(0,0);
      else
        {
          for(int i=0;i<sizes[0];++i)
            delete ptr[i];
          delete ptr;
        }
    }
#ifdef cGPU
  else (device<DEV_FPGA)
         {
           gpu_delete_tensor(gpu_device,gptr);
         }
#endif
#ifdef cFPGA
  else {
    // delete FPGA Tensor
  }
#endif
  delete tsem;
}



///////////////////////////////////////////
void Tensor::tlin(float *n)
{
  if (dim==2)
    {
      for(int i=0;i<sizes[0];++i)
        {
          int p=i*sizes[1];
          for(int j=0;j<sizes[1];++j,++p)
            n[linpos+p]=ptr2(i,j);
        }
      linpos+=tam;
    }
  else
    {
      for(int i=0;i<sizes[0];i++)
        ptr[i]->tlin(n);
    }
}


float *Tensor::toLin()
{
  if (device!=DEV_CPU) return NULL;

  float *n=(float*)malloc(tam*sizeof(float));

  linpos=0;
  if (dim==1)
    {
      for(int i=0;i<sizes[0];++i)
        n[i]=ptr1(i);
    }
  else if (dim==2)
    {
      for(int i=0;i<sizes[0];++i)
        {
          int p=i*sizes[1];
          for(int j=0;j<sizes[1];++j,++p)
            n[p]=ptr2(i,j);
        }
    }
  else
    {
      for(int i=0;i<sizes[0];i++)
        ptr[i]->tlin(n);
    }
  return n;
}


//////////////////////////////////////////////////
void Tensor::flin(float *n)
{

  if (dim==2)
    {
      for(int i=0;i<sizes[0];++i)
        {
          int p=i*sizes[1];
          for(int j=0;j<sizes[1];++j,++p)
            ptr2(i,j)=n[linpos+p];
        }
      linpos+=tam;
    }
  else
    {
      for(int i=0;i<sizes[0];i++)
        ptr[i]->flin(n);
    }
}


void Tensor::fromLin(float *n)
{
  if (device!=DEV_CPU) return;

  linpos=0;
  if (dim==1)
    {
      for(int i=0;i<sizes[0];++i)
        ptr1(i)=n[i];
    }
  else if (dim==2)
    {
      for(int i=0;i<sizes[0];++i)
        {
          int p=i*sizes[1];
          for(int j=0;j<sizes[1];++j,++p)
            ptr2(i,j)=n[p];
        }
    }
  else
    {
      for(int i=0;i<sizes[0];i++)
        ptr[i]->flin(n);
    }
}

///////////////////////////////////////////
shape Tensor::getshape()
{
  shape s=sizes;
  return s;
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

  fprintf(stderr,"Total bytes=%ld\n",tam*sizeof(float));
  if (device==DEV_CPU) fprintf(stderr,"Device=CPU\n");
  else if (device<DEV_FPGA) fprintf(stderr,"Device=GPU (%d)\n",gpu_device);
  else fprintf(stderr,"Device=FPGA\n");
}




///////////////////////////////////////////

void Tensor::print()
{

  if (device==DEV_CPU)
    {
      if (dim==1) cout<<ptr1;
      else if (dim==2) cout<<ptr2;
      else
        for(int i=0;i<sizes[0];++i)
          {
            ptr[i]->print();
            cout<<"\n";
          }
      cout<<"\n";
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      if (dim<3)
        {

          gpu_set_device(gpu_device);
          float *v= (float*)malloc(tam*sizeof(float));
          cudaMemcpy(v,gptr,tam*sizeof(float),cudaMemcpyDeviceToHost);
          if (dim==2)
            {
              int i,j,p=0;
              for(i=0;i<sizes[0];++i)
                {
                  for(j=0;j<sizes[1];++j,++p)
                    printf("%f ",v[p]);
                  printf("\n");
                }
            }
          else
            {
              int i;
              for(i=0;i<sizes[0];++i)
                printf("%f ",v[i]);
              printf("\n");
            }
          free(v);

        }
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
}


///////////////////////////////////////////
void Tensor::set(float v)
{
  if (device==DEV_CPU)
    {
      if (dim==1)
        for(int i=0;i<sizes[0];++i) ptr1(i)=v;
      else if (dim==2)
        for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2(i,j)=v;
      else
        for(int i=0;i<sizes[0];++i)
          ptr[i]->set(v);
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      gpu_set(this,v);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
}


///////////////////////////////////////////
void Tensor::mult(float v)
{
  if (device==DEV_CPU)
    {
      if (dim==1)
        for(int i=0;i<sizes[0];++i) ptr1(i)*=v;
      else if (dim==2)
        for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2(i,j)*=v;
      else
        for(int i=0;i<sizes[0];++i)
          ptr[i]->mult(v);
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      gpu_mult(this,v);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
}



///////////////////////////////////////////
void Tensor::div(float v){mult(1.0/v);}

///////////////////////////////////////////
void Tensor::sum(float v){
  if (device==DEV_CPU)
    {
      if (dim==1)
        for(int i=0;i<sizes[0];++i) ptr1(i)+=v;
      else if (dim==2)
        for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2(i,j)+=v;
      else
        for(int i=0;i<sizes[0];++i)
          ptr[i]->sum(v);
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      gpu_sum(this,v);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
}
///////////////////////////////////////////
void Tensor::sub(float v){sum(-v);}

///////////////////////////////////////////
void Tensor::set_log()
{
  if (device==DEV_CPU)
    {
      if (dim==1)
        for(int i=0;i<sizes[0];++i) ptr1(i)=log(ptr1(i));
      else if (dim==2)
        for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2(i,j)=log(ptr2(i,j));
      else
        for(int i=0;i<sizes[0];++i)
          ptr[i]->set_log();
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      gpu_log(this);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
}
///////////////////////////////////////////
void Tensor::set_exp()
{
  if (device==DEV_CPU)
    {
      if (dim==1)
        for(int i=0;i<sizes[0];++i) ptr1(i)=exp(ptr1(i));
      else if (dim==2)
        for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2(i,j)=exp(ptr2(i,j));
      else
        for(int i=0;i<sizes[0];++i)
          ptr[i]->set_exp();
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      gpu_exp(this);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
}
///////////////////////////////////////////
void Tensor::set_sqrt(){
  if (device==DEV_CPU)
    {
      if (dim==1)
        for(int i=0;i<sizes[0];++i) ptr1(i)=sqrt(ptr1(i));
      else if (dim==2)
        for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2(i,j)=sqrt(ptr2(i,j));
      else
        for(int i=0;i<sizes[0];++i)
          ptr[i]->set_sqrt();
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      gpu_sqrt(this);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
}
///////////////////////////////////////////
void Tensor::set_sqr(){
  if (device==DEV_CPU)
    {
      if (dim==1)
        for(int i=0;i<sizes[0];++i) ptr1(i)*=ptr1(i);
      else if (dim==2)
        for(int i=0;i<sizes[0];++i) for(int j=0;j<sizes[1];++j) ptr2(i,j)*=ptr2(i,j);
      else
        for(int i=0;i<sizes[0];++i)
          ptr[i]->set_sqr();
    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      gpu_sqr(this);
    }
#endif
#ifdef cFPGA
  else {

  }
#endif
}

///////////////////////////////////////////
void Tensor::rand()
{
  if (device==DEV_CPU)
    {
      if (dim==1)
        for(int i=0;i<sizes[0];++i) ptr1(i)=uniform()*0.1;
      else if (dim==2)
        {
          float s=sqrt(1.0/sizes[0]);
          for(int i=0;i<sizes[0];++i)
            for(int j=0;j<sizes[1];++j)
              ptr2(i,j)=gauss(0.0,s);
        }
      else
        for(int i=0;i<sizes[0];++i)
          ptr[i]->rand();

    }
#ifdef cGPU
  else if (device<DEV_FPGA)
    {
      //gpu_rand();
    }
#endif
#ifdef cFPGA
  else {

  }
#endif

}


///////////////////////////////////////////
