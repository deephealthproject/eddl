
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



#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../../tensor/tensor.h"
#include "tensor_cuda.h"
#include "tensor_kernels.h"

extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];


// CUDA, NVIDIA compute capabilities:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
// -----------------------------------------------------------------
//                      GRID
// Maximum dimensionality of grid of thread blocks:	3
// Maximum x-dimension of a grid of thread blocks	(2^31)-1
// Maximum y- or z-dimension of a grid of thread blocks: 65535
//                   THREAD BLOCK
// Maximum dimensionality of thread block:	3
// Maximum x- or y-dimension of a block:	1024
// Maximum z-dimension of a block:	64
//
// Maximum number of threads per block:	1024
// -----------------------------------------------------------------

// MAX THREADS PER BLOCK
#define MAX_TPB 1024

#define setDims(A) int r,c;r=(A->size/MAX_TPB);if (r==0) {r=1;c=A->size;}else {if (A->size%MAX_TPB) r++;c=MAX_TPB;}dim3 dimGrid(r);dim3 dimBlock(c);


static const char *_curandGetErrorEnum(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";
            break;
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";
            break;
        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";
            break;

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";
            break;

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";
            break;

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";
            break;

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";
            break;

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
            break;
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        default:
	    fprintf(stderr,"Not all curand errors here %d\n",error);
	    exit(-1);
    }

}

void check_cublas(cublasStatus_t status, const char *f)
{
  if ( status!=  CUBLAS_STATUS_SUCCESS)
  {
     fprintf(stderr,"Error in cublas execution in %s\n",f);
     exit(1);
  }
}

void check_curand(curandStatus_t status, const char *f)
{
  if ( status!=  CURAND_STATUS_SUCCESS)
  {
     fprintf(stderr,"Error in curand execution in %s\n",_curandGetErrorEnum(status));
     exit(1);
  }
}

///////////////////////////////////////////

void gpu_set(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  set<<<dimGrid,dimBlock>>>(A->ptr,v,r,c);
  check_cuda(cudaDeviceSynchronize(),"set");

}

void gpu_mult(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  mult<<<dimGrid,dimBlock>>>(A->ptr,v,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"mult");

}

void gpu_sum(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  sum<<<dimGrid,dimBlock>>>(A->ptr,v,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"sum");

}

void gpu_log(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  log<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"log");

}

void gpu_exp(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  exp<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"exp");

}

void gpu_sqrt(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  sqrt<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"sqrt");

}

void gpu_sqr(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  sqr<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"sqr");

}

void gpu_mask(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  mask<<<dimGrid,dimBlock>>>(A->ptr,v,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"mask");

}

void gpu_total_sum(Tensor *A,float *tot)
{
  float *total;
  int device=A->gpu_device;
  cudaSetDevice(device);
  float t=0;

  setDims(A)

  check_cuda(cudaMalloc((void**)&total,sizeof(float)),"create float in total_sum");

  check_cuda(cudaMemcpy(total,&t,sizeof(float),cudaMemcpyHostToDevice),"error copy in total_sum");

  reduce_array_sum<<<dimGrid,dimBlock>>>(A->ptr,A->size,total);

  check_cuda(cudaMemcpy(tot,total,sizeof(float),cudaMemcpyDeviceToHost),"error copy in total_sum");

  check_cuda(cudaFree(total),"delete float in total_sum");
}

///////////////////////////////////////////

void gpu_copy_to_gpu(float *nptr,Tensor *A)
{
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(A->ptr,nptr,A->size*sizeof(float),cudaMemcpyHostToDevice),"gpu_copy_to_gpu");
}

void gpu_copy_from_gpu(Tensor *A,float *nptr)
{
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(nptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToHost),"gpu_copy_to_gpu");
}

void gpu_copy_gpu(Tensor *A,Tensor *B)
{
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(B->ptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToDevice),"gpu_copy_gpu");
}

void gpu_fill(Tensor *A,int aini,int aend,Tensor *B,int bini,int bend,int inc)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  int at=A->size/A->shape[0];
  int bt=B->size/B->shape[0];

  int t=1;
  for(int i=2;i<B->ndim;i++)
    t*=B->shape[i];

  int tot=B->shape[0]*(bend-1)*B->shape[1]*t;

  int r,c;


  while (aend-aini>0) {

      if ((aend-aini)>MAX_TPB) r=MAX_TPB;
      else r=(aend-aini);
      c=t;

      dim3 dimGrid(A->shape[0],c);
      dim3 dimBlock(r);

      fill<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,t,aini,at,bini,bt,tot,inc);
      aini+=MAX_TPB;
      bini+=MAX_TPB;

  }


    //check_cuda(cudaDeviceSynchronize(),"fill");

  //}
}

///////////////////////////////////////////

void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  float alfa=1.0;
  float beta=(float)incC;

  cublasOperation_t trA = CUBLAS_OP_N;
  cublasOperation_t trB = CUBLAS_OP_N;

  int ldA=A->shape[1];
  int ldB=B->shape[1];
  int ldC=B->shape[1];
  int m=B->shape[1];
  int n=A->shape[0];
  int k=B->shape[0];


  if (tA)
  {
    trA = CUBLAS_OP_T;
  	n=A->shape[1];
  }
  if (tB)
    {
  	trB = CUBLAS_OP_T;
    m=B->shape[0];
  	k=B->shape[1];
    ldC=B->shape[0];
    }

  check_cublas(cublasSgemm(hcublas[device],trB,trA,m,n,k,&alfa,B->ptr,ldB,A->ptr,ldA,&beta,C->ptr,ldC),"mult2D");

}

void gpu_sum2D(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC)
{
    int device=A->gpu_device;
    cudaSetDevice(device);

    int m=A->shape[1];
    int n=B->shape[0];
    int ldA=A->shape[1];
    int ldB=B->shape[1];
    int ldC=A->shape[1];

    float alfa=scA;
    float beta=scB;
    float one=1.0;


    if (incC){
        check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,A->ptr,ldA,&one,C->ptr,ldB,C->ptr,ldC),"sum2D");
        check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,B->ptr,ldA,&one,C->ptr,ldB,C->ptr,ldC),"sum2D");
    }
    else
        check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,A->ptr,ldA,&beta,B->ptr,ldB,C->ptr,ldC),"sum2D");

}

void gpu_sum(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC)
{
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A)


    sum<<<dimGrid,dimBlock>>>(scA,A->ptr,scB,B->ptr,C->ptr,incC,A->size);
    check_cuda(cudaDeviceSynchronize(),"sum");
}

///////////////////////////////////////////

void gpu_el_mult(Tensor *A, Tensor *B, Tensor *C,int incC)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  el_mult<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,incC,A->shape[0],c);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");
}

void gpu_el_div(Tensor *A, Tensor *B, Tensor *C,int incC)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  el_mult<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,incC,A->shape[0],r);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");
}

///////////////////////////////////////////

void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)


  sum_mat_row<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->shape[0],A->shape[1]);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}

void gpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  sum_mat_col<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->shape[0],A->shape[1]);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}

void gpu_reduce_sum2D(Tensor *A,Tensor *B,int axis,int incB)
{

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  if (!incB) gpu_set(B,0.0);

  reduce_sum2D<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->shape[0],A->shape[1],axis);


  check_cuda(cudaDeviceSynchronize(),"reduce_sum2D");
}

///////////////////////////////////////////

void gpu_rand_uniform(Tensor *A, float v)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->ptr,A->size),"gpu_rand_uniform");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_uniform");

  gpu_mult(A,v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_uniform");

}

void gpu_rand_suniform(Tensor *A, float v)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->ptr,A->size),"gpu_rand_suniform");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_suniform");

  gpu_mult(A,2*v);
  gpu_sum(A,-v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_suniform");

}

void gpu_rand_gaussian(Tensor *A, float m,float s)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  if (A->size%2) {
    gpu_set(A,0.0);
    check_curand(curandGenerateNormal(random_generator[device],A->ptr,A->size-1,m,s),"gpu_rand_gaussian");
  }
  else
    check_curand(curandGenerateNormal(random_generator[device],A->ptr,A->size,m,s),"gpu_rand_gaussian");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_gaussian");

}

void gpu_rand_binary(Tensor *A, float v)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->ptr,A->size),"gpu_rand_binary");

  gpu_mask(A,v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_binary");

}


///////////////////////////////////////////
void gpu_cent(Tensor *A,Tensor *B,Tensor *C)
{

  int device=A->gpu_device;
  cudaSetDevice(device);
  setDims(A)

  cent<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_cent");

}

void gpu_accuracy(Tensor *A,Tensor *B,int *acc)
{
  int device=A->gpu_device;
  cudaSetDevice(device);
  int r,c;

  r=A->shape[0];
  c=A->size/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  float* max_row=gpu_create_tensor(device,r);

  int *a;
  check_cuda(cudaMalloc((void**)&a,sizeof(int)),"error cudaMalloc in accuracy");
  cudaMemset(a, 0, sizeof(int));

  accuracy<<<dimBlock,dimGrid>>>(A->ptr,B->ptr,max_row,c,r,a);
  check_cuda(cudaMemcpy(acc,a,sizeof(int),cudaMemcpyDeviceToHost),"error copy in accuracy");

  cudaFree(a);
  gpu_delete_tensor(device,max_row);

}

////////////////////////////////////

void gpu_relu(Tensor *A,Tensor *B)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  relu<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}

void gpu_d_relu(Tensor *D,Tensor *I,Tensor *PD)
{
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_relu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}

////////////////////////////////////

void gpu_softmax(Tensor *A,Tensor *B)
{
  int device=A->gpu_device;
  cudaSetDevice(device);


/*
dimBlock.x=sp->row;
 dimGrid.x=1;
 int ops = sp->col*sp->row;
int sample_ndim=sp->col;

double alfa=1;
float* auxE=NULL;
  ops=sp->row;
          auxE = makeTensor(sp->col,sp->row);
          set_sc(auxE, 0.0, sp);
  	Softmax<<<dimBlock,dimGrid>>>(E,N,auxE,sample_ndim,ops);
*/

  int r,c;

  r=A->shape[0];
  c=A->shape[1];

  dim3 dimGrid(1);
  dim3 dimBlock(r);

  float* aux=gpu_create_tensor(device,A->size);
  softmax<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,aux,c,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
  gpu_delete_tensor(device,aux);
}

void gpu_d_softmax(Tensor *D,Tensor *I,Tensor *PD)
{
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_relu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}


////////////////////////////////////
/*
Descriptor

int nk, kr, kc, kz;
int sr, sc;
int ir, ic, iz;
int r, c, z;
int padr, padc;

Tensor *I; // Input map
Tensor *ID;// Delta input map
Tensor *K; // filters
Tensor *bias; // bias
Tensor *gK;// gradient filters
Tensor *gbias;// gradient bias
Tensor *D; // Delta
Tensor *O; // Outputmap
*/

void gpu_im2col(int b,ConvolDescriptor *D,int col2im)
{
  int device=D->I->gpu_device;
  cudaSetDevice(device);

  setDims(D->gpuI)


  if (col2im)
    gpu_im2col_k<<<dimGrid,dimBlock>>>(D->ID->ptr, D->gpuI->ptr,b,D->ir,D->ic,D->iz,D->K->ptr,D->nk,D->kr,D->kc,D->O->ptr,D->r,D->c,D->sr,D->sc,D->padr,col2im);
  else
    gpu_im2col_k<<<dimGrid,dimBlock>>>(D->I->ptr, D->gpuI->ptr,b,D->ir,D->ic,D->iz,D->K->ptr,D->nk,D->kr,D->kc,D->O->ptr,D->r,D->c,D->sr,D->sc,D->padr,col2im);

  check_cuda(cudaDeviceSynchronize(),"gpu_im2col");


}

void gpu_conv2D(ConvolDescriptor *D)
{

  //fprintf(stderr,"gpu_con2D in");
  int device=D->I->gpu_device;
  cudaSetDevice(device);

  int osize=D->z*D->r*D->c;

  D->gpuK->ptr=D->K->ptr;
  D->gpuO->ptr=D->O->ptr;
  for(int b=0;b<D->I->shape[0];b++,D->gpuO->ptr+=osize){ //batch
    //fprintf(stderr,"%d\n",b);
    //I->ptr=D->I->ptr+isize;
    gpu_im2col(b,D,0);

    gpu_mult2D(D->gpuK,0,D->gpuI,1,D->gpuO,0);



  }// batch

//fprintf(stderr,"gpu_con2D out");


}


void gpu_conv2D_grad(ConvolDescriptor *D)
{

  //fprintf(stderr,"gpu_con2D in");
  int device=D->I->gpu_device;
  cudaSetDevice(device);

  int osize=D->z*D->r*D->c;

  D->gpugK->ptr=D->gK->ptr;
  D->gpuD->ptr=D->D->ptr;
  for(int b=0;b<D->I->shape[0];b++,D->gpuD->ptr+=osize){ //batch
    gpu_im2col(b,D,0);

    gpu_mult2D(D->gpuD,0,D->gpuI,0,D->gpugK,1);

  }// batch
}

void gpu_conv2D_back(ConvolDescriptor *D)
{
  //fprintf(stderr,"gpu_con2D in");
  int device=D->I->gpu_device;
  cudaSetDevice(device);

  int osize=D->z*D->r*D->c;

  D->gpuK->ptr=D->K->ptr;
  D->gpuD->ptr=D->D->ptr;
  for(int b=0;b<D->I->shape[0];b++,D->gpuD->ptr+=osize){ //batch

    gpu_mult2D(D->gpuD,1,D->gpuK,0,D->gpuI,0);

    gpu_im2col(b,D,1);

  }// batch


}






















////
////
