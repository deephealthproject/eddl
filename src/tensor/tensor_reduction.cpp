/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/profiling.h"


#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

using namespace std;

int * get_reduction_map(Tensor *A, vector<int> axis)
{
  int *redmap;

  if (axis.size()>=A->ndim)
    msg("axis must be lower than tensor dim","get_reduction_map");

  redmap=(int *)malloc(A->size*sizeof(int));

  if (redmap==nullptr)
    msg("Not enough memory indexes","get_reduction_map");


  int i,j,k,l;

  vector<int> ind;
  ind.push_back(0);
  for(i=0;i<A->ndim;i++) {
      // Check if "this" dimension is going to be reduced
      bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
      if (!isFound) {  // Dims to not be reduced...
        int s=ind.size();
        for(j=0;j<s;j++)
          for(k=0; k<A->shape[i]-1; k++)
            ind.push_back(ind[j]+(k+1)*A->stride[i]);
      }
  }

  sort(ind.begin(), ind.end());

  // reduce through axis to be reduced
  vector<vector<int>> index;
  for(i=0;i<ind.size();i++)
  {
    // get axis to be reduced
    index.push_back(vector<int>());
    index[i].push_back(ind[i]);
    for(l=0;l<A->ndim;l++) {
        // Check if "this" dimension is going to be reduced
        bool isFound = find(axis.begin(), axis.end(), l) != axis.end();
        if (isFound) {  // Dims to be reduced...
          int s=index[i].size();
          for(j=0;j<s;j++)
            for(int k=0;k<A->shape[l]-1;k++)
              index[i].push_back(index[i][j]+(k+1)*A->stride[l]);
        }
      }
  }

  int p=0;
  for(i=0;i<index.size();i++,p++) {
    for(j=0;j<index[i].size();j++)
      redmap[index[i][j]]=p;
  }

  return redmap;
}


void reduce(Tensor *A, Tensor *B,string mode,vector<int> axis,int* map)
{
  int i,j;


  if (B->ndim!=A->ndim-axis.size())
    msg("dims don't match in reduction","reduce");

  j=0;
  for(i=0;i<A->ndim;i++) {
    bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
    if (!isFound) {
      if (B->shape[j]!=A->shape[i])
       msg("shapes don't match in reduction","reduce");
      j++;
     }
  }

  PROFILING_HEADER_EXTERN(reduce);

  if (map==nullptr)
    map=get_reduction_map(A,axis);

  if (A->isCPU()) {
      cpu_reduce(A,B,mode,map);
    }
  #ifdef cGPU
  else if (A->isGPU()) {
      gpu_reduce(A,B,mode,map);
    }
  #endif
  #ifdef cFPGA
  else if (A->isFPGA()) {
    fpga_reduce(A,B,mode,map);
  }
  #endif

  PROFILING_FOOTER(reduce);
}

void reduce_mean(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce(A,B,"mean",axis,map);
}
void reduce_variance(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce(A,B,"variance",axis,map);
}
void reduce_max(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce(A,B,"max",axis,map);
}

void reduce_min(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce(A,B,"min",axis,map);
}

void reduce(Tensor *A, Tensor *B,string mode,MapReduceDescriptor *MD)
{

  PROFILING_HEADER_EXTERN(reduce);

  if (A->isCPU()) {
      cpu_reduce(A,B,mode,MD);
    }
  #ifdef cGPU
  else if (A->isGPU()) {
      gpu_reduce(A,B,mode,MD);
    }
  #endif
  #ifdef cFPGA
  else if (A->isFPGA()) {
      fpga_reduce(A,B,mode,MD);
  }
  #endif

  PROFILING_FOOTER(reduce);
}


void reduce_mean(Tensor *A, Tensor *B,MapReduceDescriptor *MD){
  reduce(A,B,"mean",MD);
}
void reduce_variance(Tensor *A, Tensor *B,MapReduceDescriptor *MD){
  reduce(A,B,"mean",MD);
}
void reduce_max(Tensor *A, Tensor *B,MapReduceDescriptor *MD){
  reduce(A,B,"mean",MD);
}
void reduce_min(Tensor *A, Tensor *B,MapReduceDescriptor *MD)
{
  reduce(A,B,"mean",MD);
}



////////////////////////////////////////////////////////
/////// REDUCE OPERATORS
////////////////////////////////////////////////////////
void reduce_op(Tensor *A, Tensor *B,string op,vector<int> axis,int* map)
{
  int i,j;

  PROFILING_HEADER_EXTERN(reduce_op);

  if (B->ndim!=A->ndim-axis.size())
    msg("dims don't match in reduction","reduce");

    j=0;
    for(i=0;i<A->ndim;i++) {
      bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
      if (!isFound) {
        if (B->shape[j]!=A->shape[i])
         msg("shapes don't match in reduction","reduce");
        j++;
       }
    }
  if (map==nullptr)
    map=get_reduction_map(A,axis);

  if (A->isCPU()) {
      cpu_reduce_op(A,B,op,map);
    }
  #ifdef cGPU
  else if (A->isGPU()) {
      gpu_reduce_op(A,B,op,map);
    }
  #endif
  #ifdef cFPGA
  else if (A->isFPGA()) {
      fpga_reduce_op(A,B,op,map);
  }
  #endif

  PROFILING_FOOTER(reduce_op);
}

void reduce_sum(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce_op(A,B,"sum",axis,map);
}
void reduce_diff(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce_op(A,B,"diff",axis,map);
}
void reduce_mult(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce_op(A,B,"mult",axis,map);
}
void reduce_div(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce_op(A,B,"div",axis,map);
}

 void reduce_op(Tensor *A, Tensor *B,string op, MapReduceDescriptor *MD)
{

  PROFILING_HEADER_EXTERN(reduce_op);

  if (A->isCPU()) {
    cpu_reduce_op(A,B,op,MD);
  }
  #ifdef cGPU
  else if (A->isGPU()) {
      gpu_reduce_op(A,B,op,MD);
    }
  #endif
  #ifdef cFPGA
    else if (A->isFPGA()) {
    fpga_reduce_op(A,B,op,MD);
  }
  #endif

  PROFILING_FOOTER(reduce_op);
}

 void reduce_sum(Tensor *A, Tensor *B,MapReduceDescriptor *MD)
{
  reduce_op(A,B,"sum",MD);
}
 void reduce_diff(Tensor *A, Tensor *B,MapReduceDescriptor *MD){
  reduce_op(A,B,"diff",MD);
}
 void reduce_mult(Tensor *A, Tensor *B,MapReduceDescriptor *MD){
  reduce_op(A,B,"mult",MD);
}
 void reduce_div(Tensor *A, Tensor *B,MapReduceDescriptor *MD){
  reduce_op(A,B,"div",MD);
}



////////////
void reduction(ReduceDescriptor *RD){

    PROFILING_HEADER_EXTERN(reduction);

    if (RD->I->isCPU()) {
      cpu_reduction(RD);
    }
    #ifdef cGPU
    else if (RD->I->isGPU())
      {
        gpu_reduction(RD);
      }
    #endif
    #ifdef cFPGA
        else {
        fpga_reduction(RD);
        }
    #endif

    PROFILING_FOOTER(reduction);
}


void reduction_back(ReduceDescriptor *RD)
{

  PROFILING_HEADER_EXTERN(reduction_back);

  if (RD->I->isCPU()) {
    cpu_reduction_back(RD);
  }
  #ifdef cGPU
  else if (RD->I->isGPU())
    {
      gpu_reduction_back(RD);
    }
  #endif
  #ifdef cFPGA
      else {
      fpga_reduction_back(RD);
      }
  #endif

  PROFILING_FOOTER(reduction_back);
}
