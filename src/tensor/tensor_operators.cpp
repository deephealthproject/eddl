
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
#include <math.h>
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
          if (fabs(A->ptr[i]-B->ptr[i])>0.001) {
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


// TODO: Review against sum
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
        Tensor::sum(1,n,1,B,B,0);
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

    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::sum");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) {
        A->info();
        B->info();
        C->info();
        msg("Incompatible dims", "Tensor::sum");
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
    else if (mode=="sum") m=1;
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
    else if (mode=="sum") m=1;
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

        //reduce sum
        for(j=0;j<sind.size();j++) {
          if (op=="sum") {
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

        //reduce sum
        for(j=0;j<sind.size();j++) {
          if (op=="sum") {
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

////////////////////////////////
//// CONVOLUTIONS
ConvolDescriptor::ConvolDescriptor() {}

ConvolDescriptor::ConvolDescriptor(int filters, const vector<int> &ks, const vector<int> &st, string p) {
    if (ks.size() != 2) { msg("Kernels must have 3 dimensions", "ConvolDescriptor::ConvolDescriptor"); }
    if (st.size() != 2) { msg("Strides must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor"); }

    // Add filters to kernel_size
    ksize = vector<int>(ks);
    ksize.insert(ksize.begin(), 1, filters);
    stride = vector<int>(st.begin(), st.end());

    if (p == "same") {
        pad.push_back(ksize[1] / 2);
        pad.push_back(ksize[2] / 2);
    } else if (p == "none") {
        pad.push_back(0);
        pad.push_back(0);
    } else msg("Incorrect padding type", "ConvolDescriptor::ConvolDescriptor");

}

ConvolDescriptor::ConvolDescriptor(const vector<int> &ks, const vector<int> &st,
                                   const vector<int> &p) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());

    if (ksize.size() != 3) msg("Kernels must have 3 dimensions", "ConvolDescriptor::ConvolDescriptor");
    if (stride.size() != 2) msg("Strides must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor");
    if (pad.size() != 2) msg("Padding must have 2 dimensions", "ConvolDescriptor::ConvolDescriptor");
}


void ConvolDescriptor::build(Tensor *A) {

    if (A->ndim != 4) msg("Tensors are not 4D", "ConvolDescriptor::build");

    I = A;

    nk = ksize[0];
    kr = ksize[1];
    kc = ksize[2];
    kz = A->shape[1];

    sr = stride[0];
    sc = stride[1];

    iz = A->shape[1];
    ir = A->shape[2];
    ic = A->shape[3];

    padr = pad[0];
    padc = pad[1];

    z = nk;
    r = (ir - kr + 2 * padr) / sr + 1;
    c = (ic - kc + 2 * padc) / sc + 1;

    if ((r <= 0) || (c <= 0))
        msg("Invalid output shape", "ConvolDescriptor::build");

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);

    // Params
    K = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    bias = new Tensor(vector<int>{nk}, I->device);
    gK = new Tensor(vector<int>{nk, kz, kr, kc}, I->device);
    gbias = new Tensor(vector<int>{nk}, I->device);

    if (I->isCPU()) {
        // mem for ptr, lowering im2col
        ptrI=get_fmem(A->shape[0] * r * c * kr * kc * kz,"ConvolDescriptor::build");
        new(&matK) Eigen::Map<Eigen::MatrixXf>(K->ptr, kr * kc * kz, nk);
        new(&matgK) Eigen::Map<Eigen::MatrixXf>(gK->ptr, kr * kc * kz, nk);
        // convolution: matC=matA*matK
    }
    #ifdef cGPU
    else if (I->isGPU()) {
      // Big tensor with all the lowering
      gpuIB=new Tensor(vector<int>{A->shape[0]*r*c,kc*kr*kz}, I->device);

      // Tensor with variable shared ptr, delete create ptr
      gpuI=new Tensor(vector<int>{r*c,kc*kr*kz}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpuI->ptr);

      gpuO=new Tensor(vector<int>{z,r*c}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpuO->ptr);
      gpuD=new Tensor(vector<int>{z,r*c}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpuD->ptr);

      gpuK=new Tensor(vector<int>{z,kc*kr*kz}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpuK->ptr);
      gpugK=new Tensor(vector<int>{z,kc*kr*kz}, I->device);
      gpu_delete_tensor(gpuI->gpu_device,gpugK->ptr);
    }
    #endif
}

void ConvolDescriptor::resize(Tensor *A)
{
    I=A;

    delete O;
    delete D;
    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);

    if (I->isCPU()) {
        delete ptrI;
        ptrI=get_fmem(A->shape[0] * r * c * kr * kc * kz,"ConvolDescriptor::build");
    }
    #ifdef cGPU
    else if (I->isGPU()) {
      delete gpuIB;
      gpuIB=new Tensor(vector<int>{A->shape[0]*r*c,kc*kr*kz}, I->device);
    }
    #endif

}

/////////////////////////////////////////////////////////////////////
//// Conv2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D(ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    D->O->tsem->lock();
    if (D->I->isCPU()) {
        cpu_conv2D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         //gpu_conv2D_old(D);
         gpu_conv2D(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->O->tsem->unlock();
}

/////////////////////////////////////////////////////////////////////
//// Conv2D Grad
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D_grad(ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    D->gK->tsem->lock();
    if (D->I->isCPU()) {
        D->gK->set(0.0);
        D->gbias->set(0.0);
        cpu_conv2D_grad(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         D->gK->set(0.0);
         D->gbias->set(0.0);
         gpu_conv2D_grad(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->gK->tsem->unlock();
}

/////////////////////////////////////////////////////////////////////
//// Conv2D Back
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::Conv2D_back(ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    D->ID->tsem->lock();
    if (D->I->isCPU()) {
        cpu_conv2D_back(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         gpu_conv2D_back(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->ID->tsem->unlock();
}


////////////////////////////////
////  POOLING

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st,
                               const vector<int> &p) {
    ksize = vector<int>(ks.begin(), ks.end());
    stride = vector<int>(st.begin(), st.end());
    pad = vector<int>(p.begin(), p.end());

    if (ksize.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (stride.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (pad.size() != 2) msg("Padding must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
}

PoolDescriptor::PoolDescriptor(const vector<int> &ks, const vector<int> &st, string p) {
    if (ks.size() != 2) msg("Pooling Kernels must have 2 dimensions", "PoolDescriptor::PoolDescriptor");
    if (st.size() != 2) msg("Strides must have 2 dimensions", "PoolDescriptor::PoolDescriptor");

    ksize = ks;
    stride = st;

    if (p == "same") {
        pad.push_back(ksize[0] / 2);
        pad.push_back(ksize[1] / 2);
    } else if (p == "none") {
        pad.push_back(0);
        pad.push_back(0);
    } else msg("Incorrect padding type", "PoolDescriptor::PoolDescriptor");
}


void PoolDescriptor::build(Tensor *A) {
    if (A->ndim != 4) msg("Tensors are not 4D", "PoolDescriptor::build");

    I = A;

    kr = ksize[0];
    kc = ksize[1];

    sr = stride[0];
    sc = stride[1];

    iz = A->shape[1];
    ir = A->shape[2];
    ic = A->shape[3];

    padr = pad[0];
    padc = pad[1];

    z = iz;
    r = (ir - kr + 2 * padr) / sr + 1;
    //if (kr%2==0) r--;
    c = (ic - kc + 2 * padc) / sc + 1;
    //if (kc%2==0) c--;

    if ((r <= 0) || (c <= 0))
        msg("Invalid output shape", "PoolDescriptor::build");

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);


}

void PoolDescriptor::resize(Tensor *A) {

    I = A;

    delete O;
    delete D;

    O = new Tensor(vector<int>{A->shape[0], z, r, c}, A->device);
    D = new Tensor(O->getShape(), A->device);
}

/////////////////////////////////////////////////////////////////////
//// MPool2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::MPool2D(PoolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::MPool2D");

    D->O->tsem->lock();
    if (D->I->isCPU()) {
        cpu_mpool2D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
        printf("mpool2d: 1-T.OP\n");
        gpu_mpool2D(D);
        printf("mpool2d: 2-T.OP\n");
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->O->tsem->unlock();
}

/////////////////////////////////////////////////////////////////////
//// MPool2D
//// Dimensions must be compatible
//// A is input 4D Tensor, Batch x Channels x Rows x Cols
//// D is a ConvolDescriptor
/////////////////////////////////////////////////////////////////////
void Tensor::MPool2D_back(PoolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::MPool2D_back");

    D->ID->tsem->lock();
    if (D->I->isCPU()) {

        cpu_mpool2D_back(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->ID->tsem->unlock();
}


////////////////////////////////
/// COST FUNCTIONS
////////////////////////////////
// Cross-Entropy: C=-(A*log(B)+(1-A)*log(1-B))
void Tensor::cent(Tensor *A, Tensor *B, Tensor *C) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::cross-entropy");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) msg("Incompatible dims", "Tensor::cross-entropy");

    C->tsem->lock();
    if (A->isCPU()) {

        for (int i = 0; i < A->size; i++) {
            C->ptr[i] = 0;
            if (A->ptr[i] != 0.0) C->ptr[i] -= A->ptr[i] * log(B->ptr[i]+0.00001);
            if (A->ptr[i] != 1.0) C->ptr[i] -= (1.0 - A->ptr[i]) * log(1.0 - B->ptr[i]+0.00001);
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_cent(A,B,C);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


////////////////////////////////
/// METRICS FUNCTIONS
////////////////////////////////
int Tensor::accuracy(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::accuracy");
    if (!eqsize(A, B)) msg("Incompatible dims", "Tensor::accuracy");
    if (A->ndim != 2) msg("Accuracy only over 2D Tensor (batch x probs)", "Tensor::Accuracy");

    int acc = 0;

    B->tsem->lock();

    if (A->isCPU()) {
        int aind, bind;

        for (int i = 0; i < A->shape[0]; i++) {
            (*A->ptr2).col(i).maxCoeff(&aind);
            (*B->ptr2).col(i).maxCoeff(&bind);
            if (aind == bind) acc++;
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_accuracy(A,B,&acc);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    B->tsem->unlock();
    return acc;

}


////////////////////////////////
/// ACTIVATIONS
////////////////////////////////
// RELU
void Tensor::ReLu(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::ReLu");
    if (!eqsize(A, B)) msg("Incompatible dims", "Tensor::ReLu");

    B->tsem->lock();
    if (A->isCPU()) {

        for (int i = 0; i < A->size; i++) {
            if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
            else B->ptr[i] = 0.0;
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
      gpu_relu(A,B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}


// RELU Derivative, always increment over parent delta
void Tensor::D_ReLu(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_ReLu");
    if ((!eqsize(D, I)) || (!eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_ReLu");

    PD->tsem->lock();
    if (D->isCPU()) {

        for (int i = 0; i < D->size; i++) {
            if (I->ptr[i] > 0.0) PD->ptr[i] = D->ptr[i];
            else PD->ptr[i] = 0.0;
        }
    }
#ifdef cGPU
    else if (D->isGPU())
      {
        gpu_d_relu(D,I,PD);

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    PD->tsem->unlock();
}


// SOFTMAX
void Tensor::Softmax(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::Softmax");
    if (!eqsize(A, B)) msg("Incompatible dims", "Tensor::Softmax");
    if (A->ndim != 2) msg("Softmax only over 2D Tensor (batch x logits)", "Tensor::Softmax");

    B->tsem->lock();

    if (A->isCPU()) {
        float max, sum;


        for (int i = 0; i < A->shape[0]; i++) {

            max = (*A->ptr2).col(i).maxCoeff();
            for (int j = 0; j < A->shape[1]; j++)
                (*B->ptr2)(j, i) = exp((*A->ptr2)(j, i) - max);

            sum = (*B->ptr2).col(i).sum();
            for (int j = 0; j < B->shape[1]; j++)
                (*B->ptr2)(j, i) = (*B->ptr2)(j, i) / sum;
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_softmax(A,B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}


// SOFTMAX DERIVATIVE
void Tensor::D_Softmax(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_Softmax");
    if ((!eqsize(D, I)) || (!eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_Softmax");
    if (D->ndim != 2) msg("D_Softmax only over 2D Tensor (batch x delta_probs)", "Tensor::D_Softmax");


    if (D->isCPU()) {
        PD->tsem->lock();

        for (int i = 0; i < D->size; i++)
            PD->ptr[i] += D->ptr[i] * (I->ptr[i] * (1.0 - I->ptr[i]));

        PD->tsem->unlock();
    }
#ifdef cGPU
    else if (D->isGPU())
      {

        Tensor *aux=new Tensor(D->getShape(),D->device);
        aux->set(1.0);
        Tensor::sum(1.0,aux,-1.0,I,aux,0);
        Tensor::el_mult(I,aux,aux,0);
        Tensor::el_mult(D,aux,PD,1);

        delete aux;
      }
#endif
#ifdef cFPGA
    else {

    }
#endif


}


///////////////////////////

//////
