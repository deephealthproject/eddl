
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

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

#include "tensor.h"
#include "../utils.h"

#include "../hardware/cpu/cpu_convol.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;

///////////////////////////////////////////
/// TENSOR OPERATIONS AS STATIC METHODS ///
///////////////////////////////////////////

int Tensor::eqsize(Tensor *A, Tensor *B) {
    if (A->ndim != B->ndim) return 0;

    for (int i = 0; i < A->ndim; i++)
        if (A->shape[i] != B->shape[i]) return 0;

    return 1;
}

int Tensor::equal(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::equal");

    if (!eqsize(A,B)) return 0;

    if (A->isCPU()) {
      for (int i = 0; i < A->size; i++)
          if (std::fabs(A->ptr[i]-B->ptr[i])>0.001) {
            fprintf(stderr,"\n>>>>>>>>>>\n");
            fprintf(stderr,"%f != %f\n",A->ptr[i],B->ptr[i]);
            return 0;
          }
    }
    #ifdef cGPU
        else if (A->isGPU())
          {
            msg("Equal only for CPU Tensors", "Tensor::equal");
          }
    #endif
    #ifdef cFPGA
        else {
          msg("Equal only for CPU Tensors", "Tensor::equal");
        }
    #endif

    return 1;
}


// Transpose
// TODO: Review correctness
void Tensor::transpose(Tensor *A, Tensor *B, vector<int> dims) {
    B->tsem->lock();
    if (!Tensor::eqsize(A, B))
        msg("Tensors with different shape", "Tensor::transpose");

    if (A->device != B->device) msg("Tensors in different devices", "Tensor::transpose");

    Tensor *N;
    if (A == B) N = new Tensor(A->getShape(), A->device);
    else N = B;


    // Copy tensor data
    if (A->isCPU()) {
        for (int i = 0; i < A->size; i++)
            N->ptr[i] = A->ptr[i];
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

    if (A == B) delete N;

}

//
// Sum all the axis of A in B
//
// TODO: Review cost (l1/l2)
void Tensor::reduceTosum(Tensor *A, Tensor *B, int axis) {
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

///////////////////////////////////////
/// Copy from A to B
//////////////////////////////////////
// TODO: Review correctness for ndim==2
void Tensor::copy(Tensor *A, Tensor *B) {

    if (!Tensor::eqsize(A, B)) {
        A->info();
        B->info();
        msg("Tensors with different shape", "Tensor::copy");
    }

    B->tsem->lock();
    if ((A->isCPU()) && (B->isCPU())) {
        for (int i = 0; i < A->size; i++)
            B->ptr[i] = A->ptr[i];

    }
#ifdef cGPU
        else if ((A->isGPU())&&(B->isGPU())) {
          gpu_copy_gpu(A,B);
        }
        else if ((A->isCPU())&&(B->isGPU()))
          {
            gpu_copy_to_gpu(A->ptr,B);
          }
        else if ((A->isGPU())&&(B->isCPU()))
          {
            gpu_copy_from_gpu(A,B->ptr);
          }
#endif
    else {
        fprintf(stderr, "(%d %d)\n", A->device, B->device);
        msg("unsupported copy between devices", "Tensor::copy");
    }
    B->tsem->unlock();
}


///////////////////////////////////////
/// Partial copy ndim=1
//////////////////////////////////////
void Tensor::fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc) {
    if (A->ndim != B->ndim)
        msg("Tensors with different shape", "Tensor::fill");

    B->tsem->lock();
    if ((A->isCPU()) && (B->isCPU())) {
        int at = A->size / A->shape[0];
        int bt = B->size / B->shape[0];

        int t = 1;
        for (int i = 2; i < A->ndim; i++)
            t *= A->shape[i];

        for (int i = 0; i < A->shape[0]; i++) {
            int ap = (i * at) + (aini * t);
            int bp = (i * bt) + (bini * t);

            for (int j = aini; j < aend; j++) {
                for (int k = 0; k < t; k++, ap++, bp++)
                    if (inc) B->ptr[bp] += A->ptr[ap];
                    else B->ptr[bp] = A->ptr[ap];
            }
        }
    }
#ifdef cGPU
        else if ((A->isGPU())&&(B->isGPU())) {
          gpu_fill(A,aini,aend,B,bini,bend,inc);
        }
#endif
    else {
        fprintf(stderr, "(%d %d)\n", A->device, B->device);
        msg("unsupported copy between devices", "Tensor::copy");
    }
    B->tsem->unlock();
}


// TODO: Review against add
void Tensor::inc(Tensor *A, Tensor *B) {

    if (!Tensor::eqsize(A, B))
        msg("Tensors with different shape", "Tensor::inc");


    if ((A->isCPU()) && (B->isCPU())) {
        B->tsem->lock();

        for (int i = 0; i < A->size; i++)
            B->ptr[i] += A->ptr[i];

        B->tsem->unlock();
    }
#ifdef cGPU
    else if ((A->isGPU())&&(B->isGPU())) {
        Tensor::sum(1,A,1,B,B,0);
    }
    else if (((A->isCPU())&&(B->isGPU()))||((A->isGPU())&&(B->isCPU())))
    {
        Tensor *n=new Tensor(B->getShape(),B->device);
        Tensor::copy(A,n);
        Tensor::add(1,n,1,B,B,0);
        delete n;
    }
#endif
    else {
        fprintf(stderr, "(%d %d)\n", A->device, B->device);
        msg("unsupported inc between devices", "Tensor::inc");
    }
}


///////////////////////////////////////
/// Select from A to B
//////////////////////////////////////
void Tensor::select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end) {

    if ((A->size / A->shape[0]) != (B->size / B->shape[0])) {
        A->info();
        B->info();
        msg("Incompatible shape", "Tensor::select");
    }

    //B->tsem->lock();
    if ((A->isCPU()) && (B->isCPU())) {
        int s = A->size / A->shape[0];

        for (int i = ini; i < end; i++) {
            int p = sind[i] * s;
            int pb = (i - ini) * s;
            for (int j = 0; j < s; j++, p++, pb++)
                B->ptr[pb] = A->ptr[p];
        }
    } else {
        msg("unsuppoted select between devices", "Tensor::select");
    }
    //B->tsem->unlock();
}


///////////////////////////////////////
/// Get sign (+-) of all values
//////////////////////////////////////
void Tensor::sign(Tensor *A, Tensor *B) {
    B->tsem->lock();

    if (!Tensor::eqsize(A, B))
        msg("Tensors with different shape", "Tensor::sign");
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::sign");

    if (A->isCPU()) {
        for (int i = 0; i < A->size; i++)
            if (A->ptr[i] < 0) B->ptr[i] = -1.0;
            else B->ptr[i] = 1.0;
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

///////////////////////////////////////
//// MULT2D C=A*B
//// tA means transpose A {0,1}
//// tB means transpose B {0,1}
//// tC 1 means C+=A*B (increment over C)
//// Dimensions and types must be compatible
//// Only for 2D Tensors
///////////////////////////////////////
void Tensor::mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {

    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::mult2D");
    if ((A->ndim != 2) || (B->ndim != 2) || (C->ndim != 2)) msg("Only 2D tensors", "Tensor::mult2D");
    if (!tA) {
        if (!tB) {
            if ((A->shape[1] != B->shape[0]) || (A->shape[0] != C->shape[0]) || (B->shape[1] != C->shape[1]))
                msg("Incompatible dims", "Tensor::mult2D");
        } else if ((A->shape[1] != B->shape[1]) || (A->shape[0] != C->shape[0]) || (B->shape[0] != C->shape[1]))
            msg("Incompatible dims", "Tensor::mult2D");
    } else {
        if (!tB) {
            if ((A->shape[0] != B->shape[0]) || (A->shape[1] != C->shape[0]) || (B->shape[1] != C->shape[1]))
                msg("Incompatible dims", "Tensor::mult2D");
        } else if ((A->shape[0] != B->shape[1]) || (A->shape[1] != C->shape[0]) || (B->shape[0] != C->shape[1]))
            msg("Incompatible dims", "Tensor::mult2D");
    }


    C->tsem->lock();

    if (A->isCPU()) {

        if (!tB) {
            if (!tA) {
                if (!incC) *(C->ptr2) = *(B->ptr2) * (*(A->ptr2));
                else *(C->ptr2) += *(B->ptr2) * (*(A->ptr2));
            } else {
                if (!incC) *(C->ptr2) = *(B->ptr2) * ((*(A->ptr2)).transpose());
                else *(C->ptr2) += *(B->ptr2) * ((*(A->ptr2)).transpose());
            }
        } else {
            if (!tA) {
                if (!incC) *(C->ptr2) = (*(B->ptr2)).transpose() * (*(A->ptr2));
                else *(C->ptr2) += (*(B->ptr2)).transpose() * (*(A->ptr2));
            } else {
                if (!incC) *(C->ptr2) = (*(B->ptr2)).transpose() * ((*(A->ptr2)).transpose());
                else *(C->ptr2) += (*(B->ptr2)).transpose() * ((*(A->ptr2)).transpose());
            }
        }
    }

#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_mult2D(A,tA,B,tB,C,incC);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// Element Mult C=A.*B
//// incC 1 means C+=A.*B (increment over C)
//// Dimensions must be compatible
///////////////////////////////////////
void Tensor::el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
    C->tsem->lock();
    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::el_mult");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) {
        A->info();
        B->info();
        C->info();
        msg("Incompatible dims", "Tensor::el_mult");
    }

    if (A->isCPU()) {
        for (int i = 0; i < A->size; i++)
            if (incC) C->ptr[i] += A->ptr[i] * B->ptr[i];
            else C->ptr[i] = A->ptr[i] * B->ptr[i];
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_el_mult(A,B,C,incC);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// Element Div C=A./B
//// incC 1 means C+=A./B (increment over C)
//// Dimensions must be compatible
///////////////////////////////////////
void Tensor::el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {

    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::el_div");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) msg("Incompatible dims", "Tensor::el_div");

    C->tsem->lock();
    if (A->isCPU()) {

        for (int i = 0; i < A->size; i++)
            if (incC) C->ptr[i] += A->ptr[i] / B->ptr[i];
            else C->ptr[i] = A->ptr[i] / B->ptr[i];
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_el_div(A,B,C,incC);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// sum C=(sca*A)+(scb*B)
//// or C+=(sca*A)+(scb*B) if incC is 1
//// Dimensions and types must be compatible
///////////////////////////////////////
void Tensor::sum(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
    int aux = 0;

    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::add");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) {
        A->info();
        B->info();
        C->info();
        msg("Incompatible dims", "Tensor::add");
    }

    C->tsem->lock();
    if (A->isCPU()) {

        for (int i = 0; i < A->size; i++)
            if (incC) C->ptr[i] += scA * A->ptr[i] + scB * B->ptr[i];
            else C->ptr[i] = scA * A->ptr[i] + scB * B->ptr[i];
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_sum(scA,A,scB,B,C,incC);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}

void Tensor::sum(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::sum(1.0, A, 1.0, B, C, 0);
}


///////////////////////////////////////
//// sum2D_rowise C=A.rowise+B
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
///////////////////////////////////////
void Tensor::sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
    if ((A->device != B->device) || (A->device != C->device))
        msg("Tensors in different devices", "Tensor::sum2D_rowwise");
    if ((A->ndim != 2) || (B->ndim != 1) || (C->ndim != 2)) msg("sum2D_rowwise dims");
    if ((!eqsize(A, C)) || (A->shape[1] != B->shape[0])) msg("Incompatible dims", "Tensor::sum2D_rowwise");

    C->tsem->lock();
    if (A->isCPU()) {
        int p = 0;
        for (int i = 0; i < A->shape[0]; i++) {
            for (int j = 0; j < A->shape[1]; j++, p++)
                C->ptr[p] = A->ptr[p] + B->ptr[j];
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_sum2D_rowwise(A,B,C);

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// sum2D_colwise C=A.colwise+B
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
///////////////////////////////////////
void Tensor::sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {
    if ((A->device != B->device) || (A->device != C->device))
        msg("Tensors in different devices", "Tensor::sum2D_colwise");
    if ((A->ndim != 2) || (B->ndim != 1) || (C->ndim != 2)) msg("sum2D_colwise dims");
    if ((!eqsize(A, C)) || (A->shape[0] != B->shape[0])) msg("Incompatible dims", "Tensor::sum2D_colwise");

    C->tsem->lock();
    if (A->isCPU()) {
        int p = 0;
        for (int i = 0; i < A->shape[0]; i++) {
            for (int j = 0; j < A->shape[1]; j++, p++)
                C->ptr[p] = A->ptr[p] + B->ptr[i];
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_sum2D_colwise(A,B,C);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// reduce_sum2D B=reduce_sum2D(A)
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
//// axis is the dimension to be sumed
///////////////////////////////////////
void Tensor::reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
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



///////////////////////////////////////
//// reductions
///////////////////////////////////////
void Tensor::reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB){
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



//// Gradient reduction
void Tensor::delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB)
{
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
///////////////////////////////////////
//// reduced operations
///////////////////////////////////////
void Tensor::reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC){
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
//// Gradient reduced operations
void Tensor::delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC)
{
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



