/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdexcept>

#include "eddl/hardware/cpu/cpu_tensor.h"

void cpu_reduce(Tensor *A, Tensor *B,string mode,int* map)
{
  int i,j,min,max,sum;
  int s=A->size/B->size;

  if (mode=="mean") {
    B->fill_(0.0);
    for(i=0;i<A->size;i++)
      B->ptr[map[i]]+=A->ptr[i];
    B->div_(s);
  }
  else if (mode=="variance") {
    Tensor *C=B->clone();
    C->fill_(0.0);
    for(i=0;i<A->size;i++)
      C->ptr[map[i]]+=A->ptr[i];
    C->div_(s);

    B->fill_(0.0);
    for(i=0;i<A->size;i++) {
      float fv=A->ptr[i]-C->ptr[map[i]];
      B->ptr[map[i]]+=fv*fv;
    }
    B->div_(s);

    delete C;
  }
  else {
    throw std::invalid_argument("mode: " + mode + " not yet implemented");
  }
}
void cpu_reduce(Tensor *A, Tensor *B,string mode,MapReduceDescriptor *MD)
{
    cpu_reduce(A,B,mode,MD->ind);
}


void cpu_reduce_op(Tensor *A, Tensor *B,string op,int* map)
{
  int i,j,min,max,sum;
  int s=A->size/B->size;

  if (op=="sum") {
    #pragma omp parallel for
    for(i=0;i<A->size;i++)
      A->ptr[i]+=B->ptr[map[i]];
  }
  else if (op=="diff"){
    #pragma omp parallel for
    for(i=0;i<A->size;i++)
      A->ptr[i]-=B->ptr[map[i]];
  }
  else if (op=="mult"){
    #pragma omp parallel for
    for(i=0;i<A->size;i++)
      A->ptr[i]*=B->ptr[map[i]];
  }
  else if (op=="div"){
    #pragma omp parallel for
    for(i=0;i<A->size;i++)
      A->ptr[i]/=B->ptr[map[i]];
  }
  else {
    throw std::invalid_argument("op: " + op + " not yet implemented");
  }
}

void cpu_reduce_op(Tensor *A, Tensor *B,string op,MapReduceDescriptor *MD)
{
  cpu_reduce_op(A,B,op,MD->ind);
}


void cpu_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
    if (axis == 0) {
        if (!incB) for (int i = 0; i < A->shape[1]; ++i) B->ptr[i] = 0;

        int p = 0;
        for (int i = 0; i < A->shape[0]; ++i) {
            for (int j = 0; j < A->shape[1]; ++j, p++)
                B->ptr[j] += A->ptr[p];
        }

    } else {
        if (!incB) for (int i = 0; i < A->shape[0]; ++i) B->ptr[i] = 0;

        int p = 0;
        for (int i = 0; i < A->shape[0]; ++i) {
            for (int j = 0; j < A->shape[1]; ++j, p++)
                B->ptr[i] += A->ptr[p];
        }
    }
}


void cpu_reduction(ReduceDescriptor *RD){

      float val,sum;
      int ind;
      int d;
      int i,j,k,l,s;

      // [MEAN]: Compute items to be reduced
      if (RD->m==0) {
          d=1;
          for(i=0;i<RD->axis.size();i++){
              d *= RD->I->shape[RD->axis[i]];
          }
      }

      //reduce
      for(i=0;i<RD->index.size();i++)
      {
          sum=0;

          for(j=0;j<RD->index[i].size();j++) {

              float v=RD->I->ptr[RD->index[i][j]];
              if (RD->m==2) {
                  if (j==0) {val=v;ind=RD->index[i][j];}
                  else if (v>val) {
                      val=v;
                      ind=RD->index[i][j];
                  }
              }
              else if (RD->m==3) {
                if (j==0) {val=v;ind=RD->index[i][j];}
                else if (v<val) {
                    val=v;
                    ind=RD->index[i][j];
                }
              }
              else sum+=v;
          }

          // set in Output
          if (RD->m<2) { // mean or sum
              if (RD->m==0) sum/=d;
              if (RD->keepdims) {
                  for(j=0;j<RD->index[i].size();j++) {
                      RD->O->ptr[RD->index[i][j]]=sum;
                    }
              }
              else RD->O->ptr[i]=sum;
          }
          else { // max or min
              if (RD->keepdims) {
                  for(j=0;j<RD->index[i].size();j++) {
                      RD->O->ptr[RD->index[i][j]]=val;
                      RD->S->ptr[RD->index[i][j]]=ind;
                  }
              }
              else {
                  RD->O->ptr[i]=val;
                  RD->S->ptr[i]=ind;
              }
          }

      }// i
  }

  void cpu_reduction_back(ReduceDescriptor *RD){

      float val,sum;
      int ind;
      int d;
      int i,j,k,l,s;

      if (RD->m==0) {
          d=1;
          for(i=0;i<RD->axis.size();i++){
              d *= RD->I->shape[RD->axis[i]];
          }
      }

      for(i=0;i<RD->index.size();i++)
        {
            if (RD->m>=2) {
                int p=RD->S->ptr[i];
                RD->ID->ptr[p]+=RD->D->ptr[i];
            }
            else {
              val=0;
              if(RD->keepdims) {
                for(j=0;j<RD->index[i].size();j++)
                  val+=RD->D->ptr[RD->index[i][j]];
              }
              else val=RD->D->ptr[i];

              if (RD->m==0) val/=d;
              for(j=0;j<RD->index[i].size();j++)
                RD->ID->ptr[RD->index[i][j]]+=val;
            }
        }//i

    }
