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
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#include "tensor_reduction.h"


#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif


using namespace std;


void reduction(ReduceDescriptor *RD){

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
                for(j=0;j<RD->index[i].size();j++)
                    RD->O->ptr[RD->index[i][j]]=sum;
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



void reduction_back(ReduceDescriptor *RD)
{

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
