
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




__global__ void maxpool2d(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padr, int padc, float* indX, float* indY) {

    long int ops = batch * orows * ocols * odepth;
    long int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_id_x < ops) {
        // output pixel at batch=ob, coord=(or,oc) at map=oz
        int orcd=orows*ocols*odepth; // out size of batch
        int orc=orows*ocols;  // out size of slice
        int ob=thread_id_x/orcd; // current batch (ib=ob)
        int bm=thread_id_x%orcd; // index in batch
        int ouz=bm/orc; // out depth (iuz=ouz)
        int our=(bm%orc)/ocols; // out row
        int ouc=(bm%orc)%ocols; // out col

        int inr = our * sr;  // in row
        int inc = ouc * sc;  // in col
        int ircd=irows*icols*idepth; // in size of batch
        int irc=irows*icols;  // in size of batch

        int min_i = -padr;
        int max_i = irows+padr-kr;
        int i = min_i + inr;  // row

        int min_j = -padc;
        int max_j = icols+padc-kc;
        int j = min_j + inc;  // column

        int b = ob;  // batch
        int k = ouz;  // depth
        int p = thread_id_x;  // index

        // Check bounds
        if (i <= max_i && j <= max_j){

            // Get maximum value in the kernel window
            float max = 0;
            for (int ki = 0; ki < kr; ki++)  // kernel_rows
                for (int kj = 0; kj < kc; kj++) {  // kernel_cols

                    // Get pixel
                    int px = j + kj;
                    int py = i + ki;
                    int pz = k;
                    float v = 0.0;

                    if (px < 0) v = 0.0;
                    else if (py < 0) v = 0.0;
                    else if (px >= icols) v = 0.0;
                    else if (py >= irows) v = 0.0;
                    else {
                        int ptr = (b * ircd) + (pz * irc) + (py * icols) + px;
                        v = I[ptr];
                    }

                    if (v > max) {
                        max = v;
                        indX[p] = j + kj;
                        indY[p] = i + ki;
                    }
                }
            O[p] = max;
        }
    }

}

__global__ void maxpool2d_back(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, int sr,int sc,int padr, int padc, float* indX, float* indY, float* D, float* ID){

    int isize=irows * icols * idepth;
    int irsize=irows * icols;

    long int ops = batch * isize;
    long int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_id_x < ops) {

        int b=thread_id_x/isize; // current batch (ib=ob)
        int bm=thread_id_x%isize; // index in batch
        int z=bm/irsize; // out depth (iuz=ouz)
        int r=(bm%irsize)/icols; // out row
        int c=(bm%irsize)%icols; // out col

        int inr = r * sr;  // in row
        int inc = c * sc;  // in col

        int min_i = -padr;
        int max_i = irows+padr-kr;
        int i = min_i + inr;  // row

        int min_j = -padc;
        int max_j = icols+padc-kc;
        int j = min_j + inc;  // column

        int p = thread_id_x;  // index

        // Check bounds
        if (i <= max_i && j <= max_j){
            int px=indX[p];
            int py=indY[p];
            int pz=z;


            if (px>=0.0 && py>=0.0 && px<icols && p<irows){
                int p=(b*isize)+(pz*irsize)+(py*icols)+px;
                ID[p]+=D[p]; // +val
            }

        }
    }

}

__global__ void  gpu_addbias_k(float *O, int batch, int r,int c,int nk,float *bias)
{
  int size=nk*r*c;
  int thread_id_x=threadIdx.x;

  int p=blockIdx.x*size+thread_id_x*r*c;
  for (int i = 0; i < r*c; i++)
     O[p+i]+=bias[thread_id_x];

}

__global__ void  gpu_deltabias_k(float *D, int batch, int r,int c,int nk,float *bias)
{
  int size=nk*r*c;
  int thread_id_x=threadIdx.x;

  int p=blockIdx.x*size+thread_id_x*r*c;
  for (int i = 0; i < r*c; i++)
    atomicAdd(&(bias[thread_id_x]),D[p+i]);

}


__global__ void gpu_im2col_k(float* I, float *ptrI,int batch,int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int pad,int col2im)
{
  long int ops=batch*orows*ocols*kr*kc*idepth;
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;


  if (thread_id_x < ops) {
    int iz,ix,iy;

    int ksize=kr*kc*idepth;

    int im=thread_id_x/(ksize*orows*ocols);
    int ioffset=im*irows*icols*idepth;

     
    int tx=thread_id_x%(ksize*orows*ocols);


    int r=tx/ksize;
    int c=tx%ksize;

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
      if (col2im)
        atomicAdd(&(I[p+ioffset]),ptrI[thread_id_x]);
      else
	ptrI[thread_id_x]=I[p+ioffset];
    }
    else
      if (!col2im)
        ptrI[thread_id_x]=0;

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

__global__ void pow(float* a, float v, long int rows, long int cols)
{
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        a[thread_id_x]=pow(a[thread_id_x], v);

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



