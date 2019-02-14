// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
// 	     Roberto Paredes Palacios, <rparedes@dsic.upv.es>
// 	     Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <initializer_list>
#include <vector>
#include <string>
#include <iostream>
#include "tensor.h"

#ifdef cGPU
#include "gpu/tensor_cuda.h"
#include "gpu/tensor_cuda_op.h"
#endif

using namespace std;

///////////////////////////////////////////
/// TENSOR OPERATIONS AS STATIC METHODS ///
///////////////////////////////////////////

int Tensor::eqsize(Tensor *A, Tensor *B)
{
    if (A->dim!=B->dim) return 0;

    for(int i=0;i<A->dim;i++)
        if (A->sizes[i]!=B->sizes[i]) return 0;

    return 1;
}


///////////////////////////////////////
/// Copy from A to B
//////////////////////////////////////
void Tensor::copy(Tensor *A, Tensor *B)
{

    if (!Tensor::eqsize(A,B))
        msg("Tensors with different sizes","Tensor::copy");
    B->tsem->lock();
    if ((A->device==DEV_CPU)&&(B->device==DEV_CPU))
    {
        if (A->dim==1) B->ptr1=A->ptr1;
        else if (A->dim==2) B->ptr2=A->ptr2;
        else for(int i=0;i<A->sizes[0];i++) Tensor::copy(A->ptr[i],B->ptr[i]);
    }
    else if ((A->device==DEV_CPU)&&(B->device>DEV_CPU))
    {
        float *nptr=A->toLin();
//gpu_copy_from(nptr,B);
        free(nptr);
    }
    else if ((A->device>DEV_CPU)&&(B->device==DEV_CPU))
    {
        float *nptr=(float*)malloc(B->tam*sizeof(float));
//gpu_copy_to(A,nptr);
        B->fromLin(nptr);
        free(nptr);
    }
    else if ((A->device!=DEV_CPU)&&(B->device!=DEV_CPU))
    {
        cout<<A->device<<"-->"<<B->device<<"\n";
        msg("unsupported copy between devices","Tensor::copy");
    }
    B->tsem->unlock();
}


void Tensor::inc(Tensor *A, Tensor *B)
{

    if (!Tensor::eqsize(A,B))
        msg("Tensors with different sizes","Tensor::copy");

    B->tsem->lock();
    if ((A->device==DEV_CPU)&&(B->device==DEV_CPU))
    {
        if (A->dim==1) B->ptr1+=A->ptr1;
        else if (A->dim==2) B->ptr2=A->ptr2;
        else for(int i=0;i<A->sizes[0];i++) Tensor::inc(A->ptr[i],B->ptr[i]);
    }
    else if ((A->device==DEV_CPU)&&(B->device>DEV_CPU))
    {
        float *nptr=A->toLin();
        //gpu_copy_from(nptr,B);
        free(nptr);
    }
    else if ((A->device>DEV_CPU)&&(B->device==DEV_CPU))
    {
        float *nptr=(float*)malloc(B->tam*sizeof(float));
        //gpu_copy_to(A,nptr);
        B->fromLin(nptr);
        free(nptr);
    }
    else if ((A->device!=DEV_CPU)&&(B->device!=DEV_CPU))
    {
        cout<<A->device<<"-->"<<B->device<<"\n";
        msg("unsupported inc between devices","Tensor::inc");
    }
    B->tsem->unlock();
}


///////////////////////////////////////
/// Select from A to B
//////////////////////////////////////
void Tensor::select(Tensor *A, Tensor *B,vector<int> sind)
{
    B->tsem->lock();
    if ((A->device==DEV_CPU)&&(B->device==DEV_CPU))
    {
        if (A->dim==1)
            for(int i=0;i<sind.size();i++)
                B->ptr1(i)=A->ptr1(sind[i]);
        else if (A->dim==2)
            for(int i=0;i<sind.size();i++)
                for(int j=0;j<B->sizes[1];j++)
                    B->ptr2(i,j)=A->ptr2(sind[i],j);
        else
            for(int i=0;i<sind.size();i++)
                Tensor::copy(A->ptr[sind[i]],B->ptr[i]);
    }
    else if ((A->device==DEV_CPU)&&(B->device>DEV_CPU))
    {
        float *nptr=A->toLin();
//gpu_copy_from(nptr,B);
        free(nptr);
    }
    else if ((A->device>DEV_GPU)&&(B->device==DEV_CPU))
    {
        float *nptr=(float*)malloc(B->tam*sizeof(float));
//gpu_copy_to(A,nptr);
        B->fromLin(nptr);
        free(nptr);
    }
    else if ((A->device!=DEV_CPU)&&(B->device!=DEV_CPU))
    {
        msg("unsuppoted copy between devices","Tensor::select");
    }
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
void Tensor::mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC)
{

    if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::mult2D");
    if ((A->dim!=2)||(B->dim!=2)||(C->dim!=2)) msg("Only 2D tensors","Tensor::mult2D");
    if (!tA)
    {
        if (!tB)
        {
            if ((A->sizes[1]!=B->sizes[0])||(A->sizes[0]!=C->sizes[0])||(B->sizes[1]!=C->sizes[1])) msg("Incompatible dims","Tensor::mult2D");
        }
        else
        if ((A->sizes[1]!=B->sizes[1])||(A->sizes[0]!=C->sizes[0])||(B->sizes[0]!=C->sizes[1])) msg("Incompatible dims","Tensor::mult2D");
    }
    else
    {
        if (!tB)
        {
            if ((A->sizes[0]!=B->sizes[0])||(A->sizes[1]!=C->sizes[0])||(B->sizes[1]!=C->sizes[1])) msg("Incompatible dims","Tensor::mult2D");
        }
        else
        if ((A->sizes[0]!=B->sizes[1])||(A->sizes[1]!=C->sizes[0])||(B->sizes[0]!=C->sizes[1])) msg("Incompatible dims","Tensor::mult2D");
    }

    C->tsem->lock();
    if (A->device==DEV_CPU)
    {
        if (!tB)
        {
            if (!tA)
            {
                if (!incC) C->ptr2=A->ptr2*B->ptr2;
                else C->ptr2+=A->ptr2*B->ptr2;
            }
            else
            {
                if (!incC) C->ptr2=A->ptr2.transpose()*B->ptr2;
                else C->ptr2+=A->ptr2.transpose()*B->ptr2;
            }
        }
        else
        {
            if (!tA)
            {
                if (!incC) C->ptr2=A->ptr2*B->ptr2.transpose();
                else C->ptr2+=A->ptr2*B->ptr2.transpose();
            }
            else
            {
                if (!incC) C->ptr2=A->ptr2.transpose()*B->ptr2.transpose();
                else C->ptr2+=A->ptr2.transpose()*B->ptr2.transpose();
            }
        }
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {
        gpu_mult2D(A,tA,B,tB,C,incC);
    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// Element Mult C=A.*B
//// incC 1 means C+=A.*B (increment over C)
//// Dimensions must be compatible
///////////////////////////////////////
void Tensor::el_mult(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC)
{
    C->tsem->lock();
    if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::el_mult");
    if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims","Tensor::el_mult");

    if (A->device==DEV_CPU)
    {
        if (A->dim==1)
        {
            if (incC) C->ptr1+=scA*A->ptr1.cwiseProduct(scB*B->ptr1);
            else C->ptr1=scA*A->ptr1.cwiseProduct(scB*B->ptr1);
        }
        else if (A->dim==2)
        {
            if (incC) C->ptr2+=scA*A->ptr2.cwiseProduct(scB*B->ptr2);
            else C->ptr2=scA*A->ptr2.cwiseProduct(scB*B->ptr2);
        }
        else
            for(int i=0;i<A->sizes[0];i++)
                Tensor::el_mult(scA,A->ptr[i],scB,B->ptr[i],C->ptr[i],incC);
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {

    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// Element Mult C=A./B
//// incC 1 means C+=A./B (increment over C)
//// Dimensions must be compatible
///////////////////////////////////////
void Tensor::el_div(float scA,Tensor *A, float scB, Tensor *B, Tensor *C,int incC)
{
    C->tsem->lock();
    if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::el_div");
    if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims","Tensor::el_div");

    if (A->device==DEV_CPU)
    {
        if (A->dim==1)
        {
            if (incC)
            {
                for(int i=0;i<A->sizes[0];i++)
                    if (A->ptr1(i)!=0) C->ptr1(i)+=(scA*A->ptr1(i))/(scB*B->ptr1(i));
            }
            else
                for(int i=0;i<A->sizes[0];i++)
                    if (A->ptr1(i)!=0) C->ptr1(i)=(scA*A->ptr1(i))/(scB*B->ptr1(i));
            else C->ptr1(i)=0.0;
        }
        else if (A->dim==2)
        {
            if (incC)
            {
                for(int i=0;i<A->sizes[0];i++)
                    for(int j=0;j<A->sizes[1];j++)
                        if (A->ptr2(i,j)!=0) C->ptr2(i,j)+=(scA*A->ptr2(i,j))/(scB*B->ptr2(i,j));
            }
            else
                for(int i=0;i<A->sizes[0];i++)
                    for(int j=0;j<A->sizes[1];j++)
                        if (A->ptr2(i,j)!=0) C->ptr2(i,j)=(scA*A->ptr2(i,j))/(scB*B->ptr2(i,j));
            else C->ptr2(i,j)=0.0;
        }
        else
            for(int i=0;i<A->sizes[0];i++)
                Tensor::el_div(scA,A->ptr[i],scB,B->ptr[i],C->ptr[i],incC);
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {

    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// sum C=(sca*A)+(scb*B)
//// or C+=(sca*A)+(scb*B) if incC is 1
//// Dimensions and types must be compatible
///////////////////////////////////////
void Tensor::sum(float scA, Tensor *A, float scB, Tensor *B, Tensor *C,int incC)
{
    int aux=0;
    C->tsem->lock();
    if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::sum");
    if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims","Tensor::sum");

    if (A->device==DEV_CPU)
    {
        if (A->dim==1)
            if (incC) C->ptr1+=(scA*A->ptr1)+(scB*B->ptr1);
        else C->ptr1=(scA*A->ptr1)+(scB*B->ptr1);
        else if (A->dim==2)
        if (incC) C->ptr2+=(scA*A->ptr2)+(scB*B->ptr2);
            else C->ptr2=(scA*A->ptr2)+(scB*B->ptr2);
        else
            for(int i=0;i<A->sizes[0];i++)
                Tensor::sum(scA,A->ptr[i],scB,B->ptr[i],C->ptr[i],incC);
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {
        //gpu_sum(scA,A,scB,B,C,incC);
    }
#endif
    C->tsem->unlock();
}


///////////////////////////////////////
//// sum2D_rowise C=A.rowise+B
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
///////////////////////////////////////
void Tensor::sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C)
{
    if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::sum2D_rowwise");
    if ((A->dim!=2)||(B->dim!=1)||(C->dim!=2)) msg("sum2D_rowwise dims");
    if ((!eqsize(A,C))||(A->sizes[1]!=B->sizes[0])) msg("Incompatible dims","Tensor::sum2D_rowwise");

    C->tsem->lock();
    if (A->device==DEV_CPU) C->ptr2=A->ptr2.rowwise()+B->ptr1;
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {
        gpu_sum2D_rowwise(A,B,C);

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
void Tensor::sum2D_colwise(Tensor *A, Tensor *B, Tensor *C)
{
    if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices","Tensor::sum2D_colwise");
    if ((A->dim!=2)||(B->dim!=1)||(C->dim!=2)) msg("sum2D_colwise dims");
    if ((!eqsize(A,C))||(A->sizes[0]!=B->sizes[0])) msg("Incompatible dims","Tensor::sum2D_colwise");

    C->tsem->lock();
    if (A->device==DEV_CPU) C->ptr2=A->ptr2.colwise()+B->ptr1.transpose();
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {
        gpu_sum2D_colwise(A,B,C);
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
void Tensor::reduce_sum2D(Tensor *A, Tensor *B, int axis,int incB)
{
    if (A->device!=B->device) msg("Tensors in different devices","Tensor::reduce_sum2D");
    if ((A->dim-1)!=B->dim) msg("Incorrect dims","Tensor::reduce_sum2D");
    if ((A->sizes[1-axis]!=B->sizes[0])) msg("Incompatible dims","Tensor::reduce_sum2D");
    B->tsem->lock();
    if (A->device==DEV_CPU)
    {
        if (axis==0)
        {
#pragma omp parallel for
            for(int i=0;i<A->sizes[1];++i)
            {
                if (!incB) B->ptr1(i)=0;
                for(int j=0;j<A->sizes[0];++j)
                    B->ptr1(i)+=A->ptr2(j,i);
            }
        }
        else
        {
#pragma omp parallel for
            for(int i=0;i<A->sizes[0];++i)
            {
                if (!incB) B->ptr1(i)=0;
                for(int j=0;j<A->sizes[1];++j)
                    B->ptr1(i)+=A->ptr2(i,j);
            }
        }
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {
        gpu_reduce_sum2D(A,B,axis,incB);
    }
#endif
    B->tsem->unlock();
}


///////////////////////////////////////
//// total_sum \sum A
///////////////////////////////////////
float Tensor::total_sum(Tensor *A)
{
    A->tsem->lock();
    if (A->device==DEV_CPU)
    {
        float sum=0.0;
        if (A->dim==1)
            sum=A->ptr1.sum();
        else if (A->dim==2)
            sum=A->ptr2.sum();
        else
            for(int i=0;i<A->sizes[0];i++)
                sum+=Tensor::total_sum(A->ptr[i]);
        return sum;
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {

    }
#endif
    A->tsem->lock();
    return 0;
}


////////////////////////////////
/// COST FUNCTIONS
////////////////////////////////
// Cross-Entropy: C=-(A*log(B)+(1-A)*log(1-B))
void Tensor::cent(Tensor *A,Tensor *B, Tensor *C)
{
    if (A->device!=B->device) msg("Tensors in different devices","Tensor::cross-entropy");
    if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims","Tensor::cross-entropy");
    C->tsem->lock();
    if (A->device==DEV_CPU)
    {
        if (A->dim==1)
        {
            for(int i=0;i<A->sizes[0];i++)
            {
                if (A->ptr1(i)!=0.0) C->ptr1(i)=-(A->ptr1(i)*log(B->ptr1(i)));
                if (A->ptr1(i)!=1.0) C->ptr1(i)=-((1.0-A->ptr1(i))*log(1.0-B->ptr1(i)));
            }
        }
        else if (A->dim==2)
        {
            for(int i=0;i<A->sizes[0];i++)
                for(int j=0;j<A->sizes[1];j++)
            {
                if (A->ptr2(i,j)!=0.0) C->ptr2(i,j)=-(A->ptr2(i,j)*log(B->ptr2(i,j)));
                if (A->ptr2(i,j)!=1.0) C->ptr2(i,j)=-((1.0-A->ptr2(i,j))*log(1.0-B->ptr2(i,j)));
            }
        }
        else
            for(int i=0;i<A->sizes[0];i++)
                Tensor::cent(A->ptr[i],B->ptr[i],C->ptr[i]);
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {

    }
#endif
    C->tsem->unlock();
}


////////////////////////////////
/// METRICS FUNCTIONS
////////////////////////////////
int Tensor::accuracy(Tensor *A,Tensor *B)
{
    if (A->device!=B->device) msg("Tensors in different devices","Tensor::accuracy");
    if (!eqsize(A,B)) msg("Incompatible dims","Tensor::accuracy");
    int acc=0;
    B->tsem->lock();
    if (A->device==DEV_CPU)
    {
        int aind,bind;
        if (A->dim==1)
        {
            A->ptr1.maxCoeff(&aind);
            B->ptr1.maxCoeff(&bind);
            if (aind==bind) return 1;
            else return 0;
        }
        else if (A->dim==2)
        {
            for(int i=0;i<A->sizes[0];i++)
            {
                A->ptr2.row(i).maxCoeff(&aind);
                B->ptr2.row(i).maxCoeff(&bind);
                if (aind==bind) acc++;
            }
        }
        else
            for(int i=0;i<A->sizes[0];i++)
                acc+=Tensor::accuracy(A->ptr[i],B->ptr[i]);
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {
        return 0;
    }
#endif
    B->tsem->unlock();
    return acc;

}


////////////////////////////////
/// ACTIVATIONS
////////////////////////////////
// RELU
void Tensor::ReLu(Tensor *A,Tensor *B)
{
    if (A->device!=B->device) msg("Tensors in different devices","Tensor::ReLu");
    if (!eqsize(A,B)) msg("Incompatible dims","Tensor::ReLu");
    B->tsem->lock();
    if (A->device==DEV_CPU)
    {
        if (A->dim==1)
        {
            B->ptr1=A->ptr1;
            for(int i=0;i<A->sizes[0];i++)
                if (A->ptr1(i)<0) B->ptr1(i)=0;
        }
        else if (A->dim==2)
        {
            B->ptr2=A->ptr2;
            for(int i=0;i<A->sizes[0];i++)
                for(int j=0;j<A->sizes[1];j++)
                    if (A->ptr2(i,j)<0) B->ptr2(i,j)=0;
        }
        else
            for(int i=0;i<A->sizes[0];i++)
                Tensor::ReLu(A->ptr[i],B->ptr[i]);
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {

    }
#endif
    B->tsem->unlock();
}


// RELU Derivative, always increment over parent delta
void Tensor::D_ReLu(Tensor *D, Tensor *I, Tensor *PD)
{
    if ((D->device!=I->device)||(D->device!=PD->device)) msg("Tensors in different devices","Tensor::D_ReLu");
    if ((!eqsize(D,I))||(!eqsize(D,PD))) msg("Incompatible dims","Tensor::D_ReLu");
    PD->tsem->lock();
    if (D->device==DEV_CPU)
    {
        if (D->dim==1)
        {
            for(int i=0;i<D->sizes[0];i++)
                if (I->ptr1(i)>0) PD->ptr1(i)+=D->ptr1(i);
        }
        else if (D->dim==2)
        {
            for(int i=0;i<D->sizes[0];i++)
                for(int j=0;j<D->sizes[1];j++)
                    if (I->ptr2(i,j)>0) PD->ptr2(i,j)+=D->ptr2(i,j);
        }
        else
            for(int i=0;i<D->sizes[0];i++)
                Tensor::D_ReLu(D->ptr[i],I->ptr[i],PD->ptr[i]);
    }
#ifdef cGPU
    else if (D->device<DEV_FPGA)
    {

    }
#endif
    PD->tsem->unlock();
}


// SOFTMAX
void Tensor::Softmax(Tensor *A,Tensor *B)
{
    if (A->device!=B->device) msg("Tensors in different devices","Tensor::Softmax");
    if (!eqsize(A,B)) msg("Incompatible dims","Tensor::Softmax");
    if (A->dim!=2)  msg("Softmax only over 2D Tensor (batch x logits)","Tensor::Softmax");
    B->tsem->lock();
    if (A->device==DEV_CPU)
    {
        float max,sum;

        for(int i=0;i<A->sizes[0];i++)
        {

            max=A->ptr2.row(i).maxCoeff();
            for(int j=0;j<A->sizes[1];j++)
                B->ptr2(i,j)=exp(A->ptr2(i,j)-max);

            sum=B->ptr2.row(i).sum();
            for(int j=0;j<B->sizes[1];j++)
                B->ptr2(i,j)=B->ptr2(i,j)/sum;
        }
    }
#ifdef cGPU
    else if (A->device<DEV_FPGA)
    {
    }
#endif
    B->tsem->unlock();
}


// SOFTMAX DERIVATIVE
void Tensor::D_Softmax(Tensor *D,Tensor *I,Tensor *PD)
{
    if ((D->device!=I->device)||(D->device!=PD->device)) msg("Tensors in different devices","Tensor::D_Softmax");
    if ((!eqsize(D,I))||(!eqsize(D,PD))) msg("Incompatible dims","Tensor::D_Softmax");
    if (D->dim!=2) msg("D_Softmax only over 2D Tensor (batch x delta_probs)","Tensor::D_Softmax");
    PD->tsem->lock();
    if (D->device==DEV_CPU)
    {
        for(int i=0;i<D->sizes[0];i++)
        {
            for(int j=0;j<D->sizes[1];j++)
                PD->ptr2(i,j)+=D->ptr2(i,j)*(I->ptr2(i,j)*(1.0-I->ptr2(i,j)));
        }
    }
    PD->tsem->unlock();

}


///////////////////////////

//////
