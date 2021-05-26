/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdexcept>

#include "eddl/descriptors/descriptors.h"
#include "eddl/tensor/tensor_reduction.h"


#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif


MapReduceDescriptor::MapReduceDescriptor(Tensor *A,vector<int> axis)
{
  ind=get_reduction_map(A,axis);
  gind=nullptr;
}

MapReduceDescriptor::~MapReduceDescriptor(){
  if (ind != nullptr) eddl_free(ind);
  ind = nullptr;
}

ReduceDescriptor::ReduceDescriptor() {}

ReduceDescriptor::ReduceDescriptor(Tensor *A, vector<int> axis, string mode, bool keepdims){
  this->axis=axis;
  this->keepdims=keepdims;
  ind=nullptr;
  factor=100;


  if (axis.size()>=A->ndim){
      msg("axis must be lower than tensor dim","ReduceDescriptor");
  }

  for(int i=0;i<axis.size();i++){
      if (axis[i]>=A->ndim) {
          throw std::runtime_error("axis " + std::to_string(axis[i]-1) + " >= dim=" + std::to_string(A->ndim-1));
      }
  }

  // Select mode (TODO: enumerations are preferred)
  if (mode=="mean") m=0;
  else if (mode=="sum") m=1;
  else if (mode=="max") m=2;
  else if (mode=="min") m=3;
  else{
      msg("Incorrect reduction mode", "ReduceDescriptor");
  }


  tshape os;

  if (keepdims){
    os=A->shape;
  }
  else {
      // Get output shape: {5, 3, 2} (axis=1) => {5, 2}
    for(int i=0;i<A->ndim;i++) {
      if (find(axis.begin(), axis.end(), i) == axis.end())
          os.push_back(A->shape[i]);
    }
  }


  int dev=A->device;

  I=A;
  O=new Tensor(os,dev);
//  D=new Tensor(os,dev);

  if ((m==2)||(m==3))
   S=new Tensor(os,dev);
  else S=nullptr;

  build_index();

}

ReduceDescriptor::~ReduceDescriptor(){
//    delete S;
//    delete[] ind;
//    delete[] red;
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
          for(int j=0;j<s;j++){
              for(int k=0; k<I->shape[i]-1; k++){
                  ind.push_back(ind[j]+(k+1)*I->stride[i]);
              }
          }
      }
  }

  sort(ind.begin(), ind.end());

  // reduce through axis to be reduced
  float max,sum;
  int imax;
  for(int i=0;i<ind.size();i++){
      // get axis to be reduced
      index.push_back(vector<int>());

      index[i].push_back(ind[i]);
      for(int l=0;l<I->ndim;l++) {
          // Check if "this" dimension is going to be reduced
          bool isFound = find(axis.begin(), axis.end(), l) != axis.end();
          if (isFound) {  // Dims to be reduced...
              int s=index[i].size();
              for(int j=0;j<s;j++){
                  for(int k=0;k<I->shape[l]-1;k++){
                      index[i].push_back(index[i][j]+(k+1)*I->stride[l]);
                  }
              }
          }
      }

    }
    //////
}


void ReduceDescriptor::resize(int b)
{
  int i;

  for(i=0;i<axis.size();i++){
    if (axis[i]==0) break;
  }


  if ((keepdims)||(i==axis.size())) {
    O->resize(b);
//    D->resize(b);
    if ((m==2)||(m==3))
      S->resize(b);
  }
  ind=nullptr;
  build_index();
}
































////
