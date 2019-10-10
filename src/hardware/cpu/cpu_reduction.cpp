//
// Created by Salva Carrión on 30/09/2019.
//

#include "cpu_hw.h"

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

void cpu_reduceTosum(Tensor *A, Tensor *B, int axis){
    for (int i = 0; i < B->size; i++)
        for (int j = 0; j < A->shape[axis]; j++)
            B->ptr[i] += A->ptr[j];
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
