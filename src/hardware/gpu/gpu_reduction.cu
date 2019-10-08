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

  int i,j,d,s,max,p;


  // [MEAN]: Compute items to be reduced
  if (RD->m==0) {
      d=1;
      for(i=0;i<RD->axis.size();i++){
          d *= RD->I->shape[RD->axis[i]];
      }
  }

  if (RD->ind==nullptr) {
    fprintf(stderr,"Mem GPU ind\n");

    RD->max=0;
    for(i=0;i<RD->index.size();i++)
       if(RD->max<RD->index[i].size()) RD->max=RD->index[i].size();
    RD->max++;
    s=RD->index.size()*RD->max;

    int *ind=(int *)malloc(s*sizeof(int));

    for(i=0;i<s;i++) ind[i]=-1;

    for(i=0;i<RD->index.size();i++) {
      p=i*max;
      for(j=0;j<RD->index[i].size();j++,p++)
        ind[p]=RD->index[i][j];
    }

    check_cuda(cudaMalloc((void**)&(RD->ind),s*sizeof(int)),"create_index");
    check_cuda(cudaMemcpy(ind,RD->ind,s*sizeof(int),cudaMemcpyHostToDevice),"copy ind");

    free(ind);
  }

  //reduce
  dim3 dimGrid(RD->index.size());
  dim3 dimBlock(1);

  reduction_kernel<<<dimGrid,dimBlock>>>(RD->I->ptr, RD->O->ptr, RD->S->ptr,RD->m, RD->keepdims,d,RD->ind,RD->max);
  check_cuda(cudaDeviceSynchronize(), "reduction_kernel");

}





void gpu_reduction_back(ReduceDescriptor *RD){
  int device=RD->I->gpu_device;

  cudaSetDevice(device);

  float val,sum;
  int ind;
  int d,i;


  // [MEAN]: Compute items to be reduced
  if (RD->m==0) {
      d=1;
      for(i=0;i<RD->axis.size();i++){
          d *= RD->I->shape[RD->axis[i]];
      }
  }

  //reduce
  //dim3 dimGrid(RD->index.size());
  //dim3 dimBlock(1024);

  //reduction_back_kernel<<<dimGrid,dimBlock>>>(RD->I->ptr, RD->O->ptr, RD->m, RD->keepdims);
  //check_cuda(cudaDeviceSynchronize(), "reduction_back_kernel");
}
