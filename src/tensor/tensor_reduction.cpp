/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "tensor.h"
#include "tensor_reduction.h"
#include "../hardware/cpu/cpu_hw.h"


#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#include "../hardware/gpu/gpu_hw.h"
#include "../hardware/gpu/nn/gpu_nn.h"
#endif


using namespace std;

int * Tensor::get_reduction_map(Tensor *A, vector<int> axis)
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


void Tensor::reduce(Tensor *A, Tensor *B,string mode,vector<int> axis,int* map)
{
  int i,j;

  if (map==nullptr)
    map=get_reduction_map(A,axis);

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

  if (A->isCPU()) {
      cpu_reduce(A,B,mode,axis,map);
    }
  #ifdef cGPU
  else if (A->isGPU()) {
      gpu_reduce(A,B,mode,axis,map);
    }
    #endif
}

void Tensor::reduce_mean(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce(A,B,"mean",axis,map);
}
void Tensor::reduce_variance(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce(A,B,"variance",axis,map);
}
void Tensor::reduce_max(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce(A,B,"max",axis,map);
}

void Tensor::reduce_min(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce(A,B,"min",axis,map);
}


////////////////////////////////////////////////////////
/////// REDUCE OPERATORS
////////////////////////////////////////////////////////
void Tensor::reduce_op(Tensor *A, Tensor *B,string op,vector<int> axis,int* map)
{
  int i,j;

  if (map==nullptr)
    map=get_reduction_map(A,axis);

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

  if (A->isCPU()) {
      cpu_reduce_op(A,B,op,axis,map);
    }
  #ifdef cGPU
  else if (A->isGPU()) {
      gpu_reduce_op(A,B,op,axis,map);
    }
  #endif
}
void Tensor::reduce_sum(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce_op(A,B,"sum",axis,map);
}
void Tensor::reduce_diff(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce_op(A,B,"diff",axis,map);
}
void Tensor::reduce_mult(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce_op(A,B,"mult",axis,map);
}
void Tensor::reduce_div(Tensor *A, Tensor *B,vector<int> axis,int* map)
{
  reduce_op(A,B,"div",axis,map);
}



////////////
void reduction(ReduceDescriptor *RD){

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

        }
    #endif
}


void reduction_back(ReduceDescriptor *RD)
{

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

      }
  #endif
}
