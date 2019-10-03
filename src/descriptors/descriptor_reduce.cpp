//
// Created by Salva Carrión on 26/09/2019.
//

#include "descriptors.h"


#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

ReduceDescriptor::ReduceDescriptor() {}

ReduceDescriptor::ReduceDescriptor(Tensor *A,vector<int> axis, string mode, bool keepdims)
{
  this->axis=axis;
  this->keepdims=keepdims;

  // Select mode
  if (mode=="mean") m=0;
  else if (mode=="sum") m=1;
  else if (mode=="max") m=2;
  else if (mode=="min") m=3;
  else
      msg("Incorrect reduction mode", "ReduceDescriptor");


  tshape os;

  /*if (find(axis.begin(), axis.end(), 0) != axis.end())
    msg("Batch axis reduction not allowed","ReduceDescriptor");
*/
  if (keepdims){
    os=A->shape;
  }
  else {
    for(int i=0;i<A->ndim;i++) {
      if (find(axis.begin(), axis.end(), i) == axis.end())
          os.push_back(A->shape[i]);
    }
  }

  int dev=A->device;

  I=A;
  O=new Tensor(os,dev);
  D=new Tensor(os,dev);

  if ((m==2)||(m==3))
   S=new Tensor(os,dev);

  build_index();

}

void ReduceDescriptor::build_index() {
  // indexes
  // get indexes for reduction
  index.clear();

  vector<int> ind;
  ind.push_back(0);
  for(int i=0;i<I->ndim;i++) {
      // Check if "this" dimension is going to be reduced
      bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
      if (!isFound) {  // Dims to not be reduced...
          int s=ind.size();
          for(int j=0;j<s;j++)
              for(int k=0; k<I->shape[i]-1; k++)
                  ind.push_back(ind[j]+(k+1)*I->stride[i]);
      }
  }

  sort(ind.begin(), ind.end());

  // reduce through axis to be reduced
  float max,sum;
  int imax;
  for(int i=0;i<ind.size();i++)
  {
      // get axis to be reduced
      index.push_back(vector<int>());

      index[i].push_back(ind[i]);
      for(int l=0;l<I->ndim;l++) {
          // Check if "this" dimension is going to be reduced
          bool isFound = find(axis.begin(), axis.end(), l) != axis.end();
          if (isFound) {  // Dims to be reduced...
              int s=index[i].size();
              for(int j=0;j<s;j++)
                  for(int k=0;k<I->shape[l]-1;k++)
                      index[i].push_back(index[i][j]+(k+1)*I->stride[l]);
          }
      }

    }
    //////



}


void ReduceDescriptor::resize(int b)
{

  O->resize(b);
  D->resize(b);
  if ((m==2)||(m==3))
    S->resize(b);

  build_index();
}
































////
