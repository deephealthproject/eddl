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

#include "gpu_kernels.h"

__global__ void reduction_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int max)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (ind[thread_id_x]!=-1) {

  int j;
  float sum=0;
  float v,val;
  int p=max*blockDim.x;
  int i;

  for(j=0;j<max && ind[p]!=-1;j++,p++) {
      v=I[ind[p]];
      if (m==2) {
          if (j==0) {val=v;i=p;}
          else if (v>val) {
              val=v;
              i=p;
          }
      }
      else if (m==3) {
        if (j==0) {val=v;i=p;}
        else if (v<val) {
            val=v;
            i=p;
        }
      }
      else sum+=v;
  }

  p=max*blockDim.x;
  // set in Output
  if (m<2) { // mean or sum
      if (m==0) sum/=d;
      if (keepdims) {
        for(j=0;j<max;j++,p++) {
            if (ind[p]==-1) break;
              O[ind[p]]=sum;
            }
      }
      else O[thread_id_x]=sum;
  }
  else { // max or min
      if (keepdims) {
        for(j=0;j<max;j++,p++) {
            if (ind[p]==-1) break;
              O[ind[p]]=val;
              S[ind[p]]=i;
          }
      }
      else {
          O[thread_id_x]=val;
          S[thread_id_x]=i;
      }
  }
  }
}


  /// backward
/*
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

      for(i=0;i<RD->index.size();i++)
        {
            if (RD->m>2) {
                if (RD->keepdims) {
                    int p=RD->S->ptr[i];
                    RD->ID->ptr[p]+=RD->D->ptr[i];
                }
                else {
                    int p=RD->S->ptr[i];
                    RD->ID->ptr[p]+=RD->D->ptr[i];
                }
            }
            else {
                if (RD->keepdims) {
                    if (RD->m==0)
                        RD->ID->ptr[i]+=RD->D->ptr[i]/d;
                    else
                        RD->ID->ptr[i]+=RD->D->ptr[i];
                }
                else {
                    for(j=0;j<RD->index[i].size();j++) {
                        if (RD->m==0)
                            RD->ID->ptr[RD->index[i][j]]+=RD->D->ptr[i]/d;
                        else
                            RD->ID->ptr[RD->index[i][j]]+=RD->D->ptr[i];
                    }
                }
            }
        }//i

    }
*/
