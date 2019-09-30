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

#include "tensor.h"
#include "../hardware/cpu/cpu_hw.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif


using namespace std;


void Tensor::reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
    ///////////////////////////////////////
    //// reduce_sum2D B=reduce_sum2D(A)
    //// Dimensions and types must be compatible
    //// A is 2D Tensor
    //// B is 1D Tensor
    //// axis is the dimension to be sumed
    ///////////////////////////////////////
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::reduce_sum2D");
    if ((A->ndim - 1) != B->ndim) msg("Incorrect dims", "Tensor::reduce_sum2D");
    if ((A->shape[1 - axis] != B->shape[0])) msg("Incompatible dims", "Tensor::reduce_sum2D");

    B->tsem->lock();
    if (A->isCPU()) {
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
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_reduce_sum2D(A,B,axis,incB);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    B->tsem->unlock();
}

void Tensor::reduceTosum(Tensor *A, Tensor *B, int axis) {
    //
    // Sum all the axis of A in B
    //
    // TODO: Review cost (l1/l2)
    B->tsem->lock();

    if (A->device != B->device) msg("Tensors in different devices", "Tensor::transpose");

    B->set(0.0);
    if (A->isCPU()) {
        for (int i = 0; i < B->size; i++)
            for (int j = 0; j < A->shape[axis]; j++)
                B->ptr[i] += A->ptr[j];
    }
#ifdef cGPU
    else if (A->isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    B->tsem->unlock();

}

void Tensor::reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB){
    ///////////////////////////////////////
    //// reductions
    ///////////////////////////////////////
    // Check device
    if (A->device != B->device){
        msg("Tensors in different devices", "Tensor::reduce");
    }

    // Check number of dimensions
    if (keepdims) {
        if (A->ndim!=B->ndim) msg("Incorrect dims keepdims", "Tensor::reduce");
    } else {
        if ((A->ndim - axis.size()) != B->ndim) msg("Incorrect dims ", "Tensor::reduce");
    }

    int i,j,k,l,s;
    int m,d;

    // Select mode
    if (mode=="mean") m=0;
    else if (mode=="add") m=1;
    else if (mode=="max") m=2;
    else
        msg("Incorrect reduction mode", "Tensor::reduce");

    // [MAX]
    if (m==2) {
        if (C==nullptr) msg("reduce max requires tensor with indexes", "Tensor::reduce");
        if (!eqsize(B,C)) msg("Incorrect sizes in reduce max", "Tensor::reduce");
    }

    // Check shapes
    if (keepdims) {
        for(i=0; i<A->ndim; i++) {
            if (A->shape[i]!=B->shape[i]) msg("Incompatible shapes", "Tensor::reduce");
        }
    } else {
        // Check A and B have the same shape ignoring axis to be reduced [Axis=(1)]: (3, 2*, 1) == (3, 1)
        j=0;
        for(i=0;i<A->ndim;i++) {
            // Check if "this" dimension is going to be reduced
            bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
            if (!isFound) {  // Dims to not be reduced, must match (2,4) => (2,4)
                if (A->shape[i]!=B->shape[j]){
                    msg("Incompatible shapes", "Tensor::reduce");
                } j++;
            }
        }
    }

    // [MEAN]: Compute items to be reduced
    if (m==0) {
        d=1;
        for(i=0;i<axis.size();i++){
            d *= A->shape[axis[i]];
        }
    }

    // If result is added to B
    if (!incB) B->set(0);

    // get indexes for reduction
    vector<int> ind;
    ind.push_back(0);
    for(i=0;i<A->ndim;i++) {
        // Check if "this" dimension is going to be reduced
        bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
        if (!isFound) {  // Dims to not be reduced...
            s=ind.size();
            for(j=0;j<s;j++)
                for(k=0; k<A->shape[i]-1; k++)
                    ind.push_back(ind[j]+(k+1)*A->stride[i]);
        }
    }



    sort(ind.begin(), ind.end());


    // reduce through axis to be reduced
    float max,sum;
    int imax;
    for(i=0;i<ind.size();i++)
    {
        // get axis to be reduced
        vector<int> sind;
        sind.push_back(ind[i]);
        for(l=0;l<A->ndim;l++) {
            // Check if "this" dimension is going to be reduced
            bool isFound = find(axis.begin(), axis.end(), l) != axis.end();
            if (isFound) {  // Dims to be reduced...
                s=sind.size();
                for(j=0;j<s;j++)
                    for(k=0;k<A->shape[l]-1;k++)
                        sind.push_back(sind[j]+(k+1)*A->stride[l]);
            }
        }

        //reduce
        sum=0;
        for(j=0;j<sind.size();j++) {
            float v=A->ptr[sind[j]];
            if (m==2) {
                if (j==0) {max=v;imax=sind[j];}
                else if (v>max) {
                    max=v;
                    imax=sind[j];
                }
            }
            else sum+=v;
        }

        // set in B
        if (m<2) {
            if (m==0) sum/=d;
            if (keepdims) {
                for(j=0;j<sind.size();j++)
                    B->ptr[sind[j]]+=sum;
            }
            else B->ptr[i]+=sum;
        }
        else {
            if (keepdims) {
                for(j=0;j<sind.size();j++) {
                    B->ptr[sind[j]]+=max;
                    C->ptr[sind[j]]=imax;
                }
            }
            else {
                B->ptr[i]+=max;
                C->ptr[i]=imax;
            }
        }

    }// i

}

void Tensor::delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB){
    //// Gradient reduction
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::delta_reduce");

    if (keepdims) {
        if (A->ndim!=B->ndim) msg("Incorrect dims keepdims", "Tensor::delta_reduce");
    }
    else
    if (A->ndim!= (B->ndim - axis.size())) msg("Incorrect dims ", "Tensor::delta_reduce");

    int i,j,k,l,s;
    int m,d;

    if (mode=="mean") m=0;
    else if (mode=="add") m=1;
    else if (mode=="max") m=2;
    else
        msg("Incorrect reduction mode", "Tensor::delta_reduce");

    if (keepdims) {
        for(i=0;i<B->ndim;i++)
            if (B->shape[i]!=A->shape[i]) {
                msg("Incompatible shapes", "Tensor::delta_reduce");
            }
    }
    else {
        j=0;
        for(i=0;i<B->ndim;i++) {
            if (find(axis.begin(), axis.end(), i) == axis.end()) {
                if (B->shape[i]!=A->shape[j])
                    msg("Incompatible shapes", "Tensor::delta_reduce");
                j++;
            }
        }
    }

    if (m==2) {
        if (C==nullptr)
            msg("delta_reduce max requires tensor with indexes", "Tensor::delta_reduce");
        if (!eqsize(A,C))
            msg("Incorrect sizes in reduce max", "Tensor::delta_reduce");
    }

    if (m==0) {
        d=1;
        for(i=0;i<axis.size();i++)
            d*=B->shape[axis[i]];
    }

    if (!incB) B->set(0);

    // get indexes for reduction
    vector<int> ind;
    ind.push_back(0);
    for(i=0;i<B->ndim;i++) {
        if (find(axis.begin(), axis.end(), i) == axis.end()) {
            s=ind.size();
            for(j=0;j<s;j++)
                for(k=0;k<B->shape[i]-1;k++)
                    ind.push_back(ind[j]+(k+1)*B->stride[i]);
        }
    }


    sort(ind.begin(), ind.end());



    for(i=0;i<A->size;i++)
    {
        vector<int> sind;
        if (!keepdims) {
            sind.push_back(ind[i]);
            for(l=0;l<B->ndim;l++) {
                if (find(axis.begin(), axis.end(), l) != axis.end()) {
                    s=sind.size();
                    for(j=0;j<s;j++)
                        for(k=0;k<B->shape[l]-1;k++)
                            sind.push_back(sind[j]+(k+1)*B->stride[l]);
                }
            }
        }

        if (m==2) {
            if (keepdims) {
                int p=C->ptr[i];
                B->ptr[p]+=A->ptr[i];
            }
            else {
                int p=C->ptr[i];
                B->ptr[p]+=A->ptr[i];
            }
        }
        else {
            if (keepdims) {
                if (m==0)
                    B->ptr[i]+=A->ptr[i]/d;
                else
                    B->ptr[i]+=A->ptr[i];
            }
            else {
                for(j=0;j<sind.size();j++) {
                    if (m==0)
                        B->ptr[sind[j]]+=A->ptr[i]/d;
                    else
                        B->ptr[sind[j]]+=A->ptr[i];
                }
            }
        }
    }//i
}

void Tensor::reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC){
    ///////////////////////////////////////
    //// reduced operations
    ///////////////////////////////////////
    // Check device
    if ((A->device != B->device)||(A->device != C->device)){
        msg("Tensors in different devices", "Tensor::reduce_op");
    }

    // Check number of dimensions
    if ((A->ndim - axis.size()) != B->ndim) msg("Incorrect dims ", "Tensor::reduce_sum");
    if (!eqsize(A,C)) msg("Incorrect dims ", "Tensor::reduce_op");

    int i,j,k,l,s;
    int m,d;


    // Check A and B have the same shape ignoring axis to be reduced [Axis=(1)]: (3, 2*, 1) == (3, 1)
    j=0;
    for(i=0;i<A->ndim;i++) {
        // Check if "this" dimension is going to be reduced
        bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
        if (!isFound) {  // Dims to not be reduced, must match (2,4) => (2,4)
            if (A->shape[i]!=B->shape[j]){
                msg("Incompatible shapes", "Tensor::reduce_op");
            } j++;
        }
    }


    // get indexes for reduction
    vector<int> ind;
    ind.push_back(0);
    for(i=0;i<A->ndim;i++) {
        // Check if "this" dimension is going to be reduced
        bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
        if (!isFound) {  // Dims to not be reduced...
            s=ind.size();
            for(j=0;j<s;j++)
                for(k=0; k<A->shape[i]-1; k++)
                    ind.push_back(ind[j]+(k+1)*A->stride[i]);
        }
    }



    sort(ind.begin(), ind.end());


    // reduce through axis to be reduced

    for(i=0;i<ind.size();i++)
    {
        // get axis to be reduced
        vector<int> sind;
        sind.push_back(ind[i]);
        for(l=0;l<A->ndim;l++) {
            // Check if "this" dimension is going to be reduced
            bool isFound = find(axis.begin(), axis.end(), l) != axis.end();
            if (isFound) {  // Dims to be reduced...
                s=sind.size();
                for(j=0;j<s;j++)
                    for(k=0;k<A->shape[l]-1;k++)
                        sind.push_back(sind[j]+(k+1)*A->stride[l]);
            }
        }

        //reduce add
        for(j=0;j<sind.size();j++) {
            if (op=="add") {
                if (!incC) C->ptr[sind[j]]=A->ptr[sind[j]]+B->ptr[i];
                else C->ptr[sind[j]]+=A->ptr[sind[j]]+B->ptr[i];
            }
            else if (op=="diff") {
                if (!incC) C->ptr[sind[j]]=A->ptr[sind[j]]-B->ptr[i];
                else C->ptr[sind[j]]+=A->ptr[sind[j]]-B->ptr[i];
            }
            else {
                msg("Incorrect operation","Tensor::reduce_op");
            }
        }
    }// i

}

void Tensor::delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC)
{
    //// Gradient reduced operations
    if ((A->device != B->device)||(A->device != C->device)){
        msg("Tensors in different devices", "Tensor::reduce_op");
    }

    // Check number of dimensions
    if ((A->ndim - axis.size()) != B->ndim) msg("Incorrect dims ", "Tensor::delta_reduce_op");
    if (!eqsize(B,C)) msg("Incorrect dims ", "Tensor::delta_reduce_op");

    int i,j,k,l,s;
    int m,d;


    j=0;
    for(i=0;i<B->ndim;i++) {
        if (find(axis.begin(), axis.end(), i) == axis.end()) {
            if (B->shape[i]!=A->shape[j])
                msg("Incompatible shapes", "Tensor::delta_reduce");
            j++;
        }
    }
    // get indexes for reduction
    vector<int> ind;
    ind.push_back(0);
    for(i=0;i<A->ndim;i++) {
        // Check if "this" dimension is going to be reduced
        bool isFound = find(axis.begin(), axis.end(), i) != axis.end();
        if (!isFound) {  // Dims to not be reduced...
            s=ind.size();
            for(j=0;j<s;j++)
                for(k=0; k<A->shape[i]-1; k++)
                    ind.push_back(ind[j]+(k+1)*A->stride[i]);
        }
    }



    sort(ind.begin(), ind.end());


    if (!incC) C->set(0.0);
    // reduce through axis to be reduced
    for(i=0;i<ind.size();i++)
    {
        // get axis to be reduced
        vector<int> sind;
        sind.push_back(ind[i]);
        for(l=0;l<A->ndim;l++) {
            // Check if "this" dimension is going to be reduced
            bool isFound = find(axis.begin(), axis.end(), l) != axis.end();
            if (isFound) {  // Dims to be reduced...
                s=sind.size();
                for(j=0;j<s;j++)
                    for(k=0;k<A->shape[l]-1;k++)
                        sind.push_back(sind[j]+(k+1)*A->stride[l]);
            }
        }

        //reduce add
        for(j=0;j<sind.size();j++) {
            if (op=="add") {
                C->ptr[i]+=A->ptr[sind[j]];
            }
            else if (op=="diff") {
                C->ptr[i]+=A->ptr[sind[j]];
            }
            else {
                msg("Incorrect operation","Tensor::delta_reduce_op");
            }
        }
    }// i

}

