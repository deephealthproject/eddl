/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"



void gpu_reduction(ReduceDescriptor *RD){
  int device=RD->I->gpu_device;

  cudaSetDevice(device);

  int i,j,d,s,p;


  // [MEAN]: Compute items to be reduced
  if (RD->m==0) {
      d=1;
      for(i=0;i<RD->axis.size();i++){
          d *= RD->I->shape[RD->axis[i]];
      }
  }

  if (RD->ind==nullptr) {
    RD->red_size=RD->index[0].size();
    s=RD->index.size()*RD->red_size;

    int *ind=(int *)malloc(s*sizeof(int));

    for(i=0;i<RD->index.size();i++) {
      p=i*RD->red_size;
      for(j=0;j<RD->index[i].size();j++,p++)
        ind[p]=RD->index[i][j];
    }

    if (RD->m<2) RD->S=RD->O;

    check_cuda(cudaMalloc((void**)&(RD->ind),s*sizeof(int)),"create_index");
    check_cuda(cudaDeviceSynchronize(), "create ind");

    check_cuda(cudaMemcpy(RD->ind,ind,s*sizeof(int),cudaMemcpyHostToDevice),"copy ind");
    check_cuda(cudaDeviceSynchronize(), "copy");

    free(ind);
  }


  if (RD->m<-1) {// mean or sum
    RD->O->set(0.0);
    dim3 dimGrid(RD->red_size);
    dim3 dimBlock(RD->index.size());

    printf("KERNEL %dx%d\n",dimGrid.x,dimBlock.x);

    reduction_kernel_sum<<<dimGrid,dimBlock>>>(RD->I->ptr, RD->O->ptr, RD->m, d,RD->ind,RD->red_size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");

    if (RD->keepdims) {
      reduction_kernel_keep<<<dimGrid,dimBlock>>>(RD->O->ptr, RD->O->ptr, RD->m,d,RD->ind,RD->red_size);
      check_cuda(cudaDeviceSynchronize(), "reduction_kernel");
    }
  }else{ // still slow for max, min on conv
    RD->O->set(0.0);
    dim3 dimGrid(RD->index.size());
    dim3 dimBlock(1);
    reduction_kernel<<<dimGrid,dimBlock>>>(RD->I->ptr, RD->O->ptr, RD->S->ptr,RD->m, RD->keepdims,d,RD->ind,RD->red_size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");
  }

}


//////
////// back
//////
void gpu_reduction_back(ReduceDescriptor *RD){
  int device=RD->I->gpu_device;

  cudaSetDevice(device);

  int d,i;

  // [MEAN]: Compute items to be reduced
  if (RD->m==0) {
      d=1;
      for(i=0;i<RD->axis.size();i++){
          d *= RD->I->shape[RD->axis[i]];
      }
  }

  //reduce
  if (RD->m<-1) {// mean or sum
    dim3 dimGrid(RD->red_size);
    dim3 dimBlock(RD->index.size());



    Tensor *aux=new Tensor(RD->ID->getShape(),RD->ID->device);
    Tensor::copy(RD->ID,aux);
    RD->ID->set(0.0);

    if (RD->keepdims) {
      reduction_kernel_sum<<<dimGrid,dimBlock>>>(RD->D->ptr, RD->ID->ptr,RD->m, d,RD->ind,RD->red_size);
      check_cuda(cudaDeviceSynchronize(), "reduction_kernel");
      reduction_kernel_keep<<<dimGrid,dimBlock>>>(RD->ID->ptr,RD->ID->ptr, RD->m, d,RD->ind,RD->red_size);
      check_cuda(cudaDeviceSynchronize(), "reduction_kernel");
    }
    else {
      reduction_kernel_keep<<<dimGrid,dimBlock>>>(RD->D->ptr,RD->ID->ptr, RD->m, d,RD->ind,RD->red_size);
      check_cuda(cudaDeviceSynchronize(), "reduction_kernel");
    }



    Tensor::inc(aux,RD->ID);
    delete aux;

  }else{ // still slow for max, min on conv
    dim3 dimGrid(RD->index.size());
    dim3 dimBlock(1);
    reduction_back_kernel<<<dimGrid,dimBlock>>>(RD->D->ptr, RD->ID->ptr, RD->S->ptr,RD->m, RD->keepdims,d,RD->ind,RD->red_size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");
  }
}
