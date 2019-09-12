
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
//      Juan Maroñas: jmaronas@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////



#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>


__global__ void conv2D(float* I, int batch,int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int pad)
{
 long int ops=batch*orows*ocols*nk;
 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < ops) {
   // output pixel at batch=ob, coord=(or,oc) at map=oz
   int rcd=orows*ocols*nk;
   int rc=orows*ocols;

   int ob=thread_id_x/rcd;
   int bm=thread_id_x%rcd;

   int ouz=bm/rc;
   int our=(bm%rc)/ocols;
   int ouc=(bm%rc)%ocols;

   //
   int ircd=irows*icols*idepth;
   int irc=irows*icols;

   int kr2=kr/2;
   int kc2=kc/2;
   int krc=kr*kc;
   int ptrI;

   // Select filter outz from nk
   int ptrKb=ouz*kr*kc*idepth;

   // Convol
   float sum=0.0;
   int inr=(our*sr)+kr2-pad;
   int inc=(ouc*sc)+kr2-pad;
   for(int i=inr-kr2;i<=inr+kr2;i++) {
     if ((i>=0)&&(i<irows)) {
       for(int j=inc-kc2;j<=inc+kc2;j++,ptrKb++){
          if ((j>=0)&&(j<icols)) {
            ptrI=ob*ircd;
            ptrI+=i*icols;
            ptrI+=j;
            int ptrK=ptrKb;
            for(int k=0;k<idepth;k++) {
              sum+=I[ptrI]*K[ptrK];
              ptrI+=irc;
              ptrK+=krc;
            }// k
         }// if j
       } //j
     } //if i
     else ptrKb+=kc;
   }//i

   O[thread_id_x]=sum;

 }

}

__global__ void gpu_im2col_k(float* I, float *ptrI,int b,int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int pad,int col2im)
{
  long int ops=orows*ocols*kr*kc*idepth;
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;


  if (thread_id_x < ops) {
    int iz,ix,iy;

    int ksize=kr*kc*idepth;
    int isize=b*irows*icols*idepth;
    //int kr2=kr/2;
    //int kc2=kc/2;

    int r=thread_id_x/ksize;
    int c=thread_id_x%ksize;

    int oy=r/ocols;
    int ox=r%ocols;


    ix=(ox*sc)-pad;
    iy=(oy*sr)-pad;
    iz=c/(kr*kc);



    c=c%(kr*kc);

    iy+=c/kc;
    ix+=c%kc;


    if ((ix>=0)&&(ix<icols)&&(iy>=0)&&(iy<irows)) {
      int p=iz*(irows*icols)+(iy*icols)+ix;
      if (col2im) //I[p+isize]+=ptrI[thread_id_x];
        atomicAdd(&(I[p+isize]),ptrI[thread_id_x]);
      else ptrI[thread_id_x]=I[p+isize];
    }
    else
      if (!col2im) ptrI[thread_id_x]=0;

  }

}



__global__ void fill(float *aptr,float *bptr,int t,int aini,int at,int bini,int bt,int tot,int inc)
{
  int i=blockIdx.x;
  int j=threadIdx.x;
  int k=blockIdx.y;

  int ap=(i*at)+((aini+j)*t)+k;
  int bp=(i*bt)+((bini+j)*t)+k;

  if (bp<tot)
    if (inc) bptr[bp]+=aptr[ap];
    else bptr[bp]=aptr[ap];

}

///////////////////////////////////////////

__global__ void set(float* a, float v, long int rows, long int cols)
{
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        a[thread_id_x]=v;

}

__global__ void mult(float* a, float v, long int rows, long int cols)
{
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        a[thread_id_x]*=v;

}

__global__ void sum(float* a, float v, long int rows, long int cols)
{
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        a[thread_id_x]+=v;

}

///////////////////////////////////////////

__global__ void log(float* a, long int rows, long int cols)
{
 long int ops=rows*cols;
 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < ops)
   a[thread_id_x]=logf(a[thread_id_x]);

}

__global__ void exp(float* a, long int rows, long int cols)
{
 long int ops=rows*cols;
 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < ops)
   a[thread_id_x]=expf(a[thread_id_x]);

}

__global__ void sqrt(float* a, long int rows, long int cols)
{
 long int ops=rows*cols;
 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < ops)
   a[thread_id_x]=sqrt(a[thread_id_x]);

}

__global__ void sqr(float* a, long int rows, long int cols)
{
 long int ops=rows*cols;
 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < ops)
   a[thread_id_x]*=a[thread_id_x];

}

__global__ void mask(float* a, float v, long int rows, long int cols)
{
 long int ops=rows*cols;
 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < ops)
   a[thread_id_x]=a[thread_id_x]<v;

}

///////////////////////////////////////////

__global__ void reduce_array_sum(float* a, long int ops, float* result)
{
  long int thread_id_x = threadIdx.x+(blockDim.x*blockIdx.x);

  if (thread_id_x < ops){
    atomicAdd(result,a[thread_id_x]);
  }
}

///////////////////////////////////////////

__global__ void sum(float scA,float* a,float scB,float *b, float *c,long int incC, long int size)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size) {
        if (incC) c[thread_id_x]+=scA*a[thread_id_x]+scB*b[thread_id_x];
        else c[thread_id_x]=scA*a[thread_id_x]+scB*b[thread_id_x];
    }
}

__global__ void el_mult(float* a, float *b, float *c, long int incC, long int rows, long int cols)
{
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        if (incC) c[thread_id_x]+=a[thread_id_x]*b[thread_id_x];
        else c[thread_id_x]=a[thread_id_x]*b[thread_id_x];
}

__global__ void el_div(float* a, float *b, float *c, long int incC, long int rows, long int cols)
{
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        if (incC) c[thread_id_x]+=a[thread_id_x]/(b[thread_id_x]);
        else c[thread_id_x]=a[thread_id_x]/(b[thread_id_x]);
}

///////////////////////////////////////////

__global__ void sum_mat_row(float* a, float* b, float* c, long int rows, long int cols)
{
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        c[thread_id_x]=a[thread_id_x]+b[thread_id_x%cols];

}

__global__ void sum_mat_col(float* a, float* b, float* c, long int rows, long int cols)
{
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        c[thread_id_x]=a[thread_id_x]+b[thread_id_x/cols];

}

///////////////////////////////////////////

__global__ void reduce_sum2D(float *a,float *b,long int rows,long int cols,long int axis)
{
  long int ops=rows*cols;
  long int thread_id_x = threadIdx.x+(blockDim.x*blockIdx.x);

  if (thread_id_x < ops){
    if (axis==0)
      atomicAdd(&(b[thread_id_x%cols]),a[thread_id_x]);
    else
      atomicAdd(&(b[thread_id_x/cols]),a[thread_id_x]);
  }

}

///////////////////////////////////////////

__global__ void cent(float* a, float* b, float* c, long int size)
{

 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < size){
   c[thread_id_x]=0;
   if (a[thread_id_x]) c[thread_id_x]-=a[thread_id_x]*logf(b[thread_id_x]+0.00001);
   if (a[thread_id_x]!=1.0) c[thread_id_x]-=(1.0-a[thread_id_x])*logf(1.0-b[thread_id_x]+0.00001);
  }
}

__global__ void accuracy(float* T, float* N,float* acc,long int cols, long int total_ops, int* MC_err)
{

long int thread_id_x = threadIdx.x + blockIdx.x*blockDim.x;
long int result_t=T[thread_id_x*cols];
float result_n=N[thread_id_x*cols];

long int row_max_t=0;
long int row_max_n=0;

long int aux_t;
float aux_n;
if (thread_id_x < total_ops)
{
  for(long int i=1;i<cols;i++)
  {
   aux_t=T[thread_id_x*cols+i];
   aux_n=N[thread_id_x*cols+i];

	if (aux_t>result_t)
	 {
  		result_t=aux_t;
      row_max_t=i;
   }
  if (aux_n>result_n)
	 {
		result_n=aux_n;
    row_max_n=i;
   }
  }

  acc[thread_id_x]=row_max_t;
  atomicAdd(MC_err,(long int)(row_max_t==row_max_n));
}

}

///////////////////////////////////////////

__global__ void relu(float *a,float *b,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (a[thread_id_x]>0.0) b[thread_id_x]=a[thread_id_x];
    else b[thread_id_x]=0.0;
   }
}

__global__ void d_relu(float *d,float *i,float *pd,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (i[thread_id_x]>0.0) pd[thread_id_x]=d[thread_id_x];
    else pd[thread_id_x]=0.0;
   }

}

__global__ void softmax(float* E,float* N,float* auxE ,long int sample_ndim, long int n_vals)
{
    float C_value=0;
    long int thread_id_x = threadIdx.x + blockIdx.x*blockDim.x;
    float maxCoef = E[thread_id_x*sample_ndim];
    float actualCoef = 0;
    if (thread_id_x<n_vals)
    {

	    for (long int cA = 1; cA < sample_ndim; cA++)
    		if (E[thread_id_x*sample_ndim+cA] > maxCoef)
    			 maxCoef=E[thread_id_x*sample_ndim+cA];

	    for (long int cA = 0; cA < sample_ndim; cA++)
  		{
  			actualCoef=expf(E[thread_id_x*sample_ndim+cA]-maxCoef);
  			auxE[thread_id_x*sample_ndim+cA]=actualCoef;
        C_value+=actualCoef;
  		}

      for (long int cA=0; cA < sample_ndim; cA++)
	       N[thread_id_x*sample_ndim+cA]=auxE[thread_id_x*sample_ndim+cA]/C_value;
    }

}
























///////////////////////////////////////////


///////////////////////////////////////////


///////////////////////////////////////////
