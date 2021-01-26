/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/layers/reductions/layer_reductions.h"
#include "eddl/layers/operators/layer_operators.h"

using namespace std;

int LBatchNorm::total_layers = 0;


LBatchNorm::LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    input=parent->output;
    isnorm=true;

    shape.push_back(input->shape[1]);

    if ((input->ndim != 2)&&(input->ndim != 4)) {
        input->info();
        msg("LBatchNorm only works over 1D (Dense) or 2D (Conv) tensors","LBatchNorm");
    }

    if(name.empty()) this->name = "batchnorm" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    output=new Tensor(input->getShape(),dev);
    opa=new Tensor(input->getShape(),dev);

    mean=new Tensor(shape,dev);
    mean->fill_(0.0);
    variance=new Tensor(shape,dev);
    variance->fill_(1.0);

    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);

    if (affine) {

        bn_g=new Tensor(shape,dev);
        bn_b=new Tensor(shape,dev);

        gbn_g=new Tensor(shape,dev);
        gbn_b=new Tensor(shape,dev);

        params.push_back(bn_g);
        params.push_back(bn_b);

        gradients.push_back(gbn_g);
        gradients.push_back(gbn_b);
    }
#ifdef cCUDNN
      cudnn_handle = hdnn;
      data_type = CUDNN_DATA_FLOAT;
      tensor_format = CUDNN_TENSOR_NCHW;
      bn_mode =(input->ndim > 2) ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
      cudnnCreateTensorDescriptor(&xDesc);
      cudnnStatus_t bbb = cudnnSetTensor4dDescriptor(xDesc, tensor_format, data_type,
                 input->shape[0],input->shape[1],
                 (input->shape.size()>2) ? input->shape[2] : 1,
                 (input->shape.size()>3) ? input->shape[3] : 1);
      if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"Error create bn tensor descriptor x "<< cudnnGetErrorString(bbb) <<std::endl;
      cudnnCreateTensorDescriptor(&yDesc);
      bbb = cudnnSetTensor4dDescriptor(yDesc, tensor_format, data_type,
                 output->shape[0],output->shape[1],
                 (output->shape.size()>2) ? output->shape[2] : 1,
                 (output->shape.size()>3) ? output->shape[3] : 1);
      if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"Error create bn tensor descriptor y "<< cudnnGetErrorString(bbb) <<std::endl;
      cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
      if(bn_mode == CUDNN_BATCHNORM_SPATIAL){
      bbb = cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, tensor_format, data_type,
                 1,output->shape[1],
                 1, 1);
      }
      else{
      bbb = cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, tensor_format, data_type,
                 1,output->shape[1],
                 (output->shape.size()>2) ? output->shape[2] : 1,
                 (output->shape.size()>3) ? output->shape[3] : 1);

      }
      exponentialAverageFactor = 1.0 - this->momentum;
#endif

    // no trainable:
    params.push_back(mean);
    params.push_back(variance);

    parent->addchild(this);
    addparent(parent);
}

LBatchNorm::~LBatchNorm(){
    delete bn_mean;
    delete bn_var;
    delete opa; //output pre-affine
}

// override functions:
int LBatchNorm::get_trainable_params_count()
{
    if (affine) return 2;  // only 2 trainable params
    else return 0;  // no trainable params
}

void LBatchNorm::initialize() {
    if (affine) {
        params[0]->fill_(1.0);
        params[1]->fill_(0.0);
    }
}

void LBatchNorm::resize(int batch){
    if (batch!=output->shape[0]) {
        opa->reshape_(output->getShape());
        output->resize(batch);
        opa->resize(batch);
#ifdef cCUDNN
        cudnnStatus_t bbb = cudnnSetTensor4dDescriptor(xDesc, tensor_format, data_type,
                                                       batch,input->shape[1],
                                                       (input->shape.size()>2) ? input->shape[2] : 1,
                                                       (input->shape.size()>3) ? input->shape[3] : 1);
        if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"Error create bn tensor descriptor x "<< cudnnGetErrorString(bbb) <<std::endl;
        bbb = cudnnSetTensor4dDescriptor(yDesc, tensor_format, data_type,
                                         batch,output->shape[1],
                                         (output->shape.size()>2) ? output->shape[2] : 1,
                                         (output->shape.size()>3) ? output->shape[3] : 1);
        if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"Error create bn tensor descriptor y "<< cudnnGetErrorString(bbb) <<std::endl;
        /*bbb = cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, tensor_format, data_type,
                                         batch,output->shape[1],
                                         (output->shape.size()>2) ? output->shape[2] : 1,
                                         (output->shape.size()>3) ? output->shape[3] : 1);*/
#endif
    }
}

// Batchnorm works over 2D Tensors
// Essentialy 4D Tensors are reshaped as 2D and
// Permute 4D tensors and set N,M values.
void LBatchNorm::forward() {
    // Input = Output = opa = {Batch,Channels,H,W} OR {Batch,Dim}
    // bn_mean = bn_var = mean = variance = bn_g = bn_b = {Channels} or {Dim}

#ifndef cCUDNN
    int M,N;
    int b,z,r,c,d;
    Tensor *in;

    if (input->ndim==2) {
        N=b=input->shape[0];
        M=d=input->shape[1];
        in=input->clone();
    }
    else {
        b=input->shape[0];
        M=z=input->shape[1];
        r=input->shape[2];
        c=input->shape[3];
        N=b*r*c;

        in=new Tensor({b*r*c*z},input->device);
        tensorNN::permute_channels_last(input,in);
        in->reshape_({N,M});
        opa->reshape_({N,M});
    }

    BN_forward(in,bn_mean,bn_var,mean,variance,momentum,epsilon,mode==TRMODE);


    Tensor::copy(in,opa);
    if (affine) {
        Tensor *var=new Tensor({N,M},input->device);
        Tensor *ones=new Tensor({N,1},input->device);
        ones->fill_(1.0);

        // apply affine transform in=gamma*in+beta
        rmult(in,bn_g,ones,var);
        rsum(in,bn_b,ones,var);
        delete var;
        delete ones;
    }

    // copy in to ouput
    if (input->ndim==4) {
        tensorNN::permute_channels_first(in,output);
    }
    else Tensor::copy(in,output);


    delete in;
#else
    float alpha = 1.0;
    float beta = 0.0;

    cudnnStatus_t nnn=cudnnBatchNormalizationForwardTraining(cudnn_handle, bn_mode, &alpha, &beta,
                                                             xDesc, input->ptr, yDesc, output->ptr,
                                                             bnScaleBiasMeanVarDesc, bn_g->ptr, bn_b->ptr,
                                                             exponentialAverageFactor, mean->ptr, variance->ptr, epsilon,
                                                             bn_mean->ptr, bn_var->ptr);
    if(nnn != CUDNN_STATUS_SUCCESS) std::cout<<"Error fwd BN  "<< cudnnGetErrorString(nnn) <<std::endl;
#endif

}

void LBatchNorm::backward(){

    //std::cout<<"BN layer BWD: "<< this->name <<std::endl;
#ifndef cCUDNN
    int M,N;
    int b,z,r,c,d;

    Tensor *dp;

    if (input->ndim==2) {
        N=b=input->shape[0];
        M=d=input->shape[1];

        dp=delta->clone();
    }
    else {
        b=input->shape[0];
        M=z=input->shape[1];
        r=input->shape[2];
        c=input->shape[3];

        N=b*r*c;

        // permute input and delta
        dp=new Tensor({b,r,c,z},input->device);

        tensorNN::permute_channels_last(delta,dp);

        dp->reshape_({N,M});

    }

    // Affine
    if (affine) {
        Tensor *A=new Tensor({N,M},delta->device);
        Tensor *ones=new Tensor({N},delta->device);
        ones->fill_(1.0);
        Tensor *m=new Tensor({1,M},delta->device);
        //1 gamma
        Tensor::el_mult(dp,opa,A,0);
        cmean(A,m,ones);
        Tensor::add(1,gbn_g,1,m,gbn_g,1);

        //2 Beta
        cmean(dp,m,ones);
        Tensor::add(1,gbn_b,1,m,gbn_b,1);

        // delta=dE/dY
        // Obtain dE/dY from delta:
        rmult(dp,bn_g,ones,A);
        delete A;
        delete ones;
        delete m;
    }

    BN_backward(dp,bn_var,opa);

    // Inc parent delta
    if (input->ndim==4) {
        tensorNN::permute_channels_first(dp,delta);
        Tensor::inc(delta, parent[0]->delta);
    }
    else Tensor::inc(dp, parent[0]->delta);

    delete dp;
#else
      float alphaDataDiff = 1.0;
      float betaDataDiff = 0.0;
      float alphaParamDiff = 1.0;
      float betaParamDiff = 0.0;


      cudnnStatus_t nnn= cudnnBatchNormalizationBackward(cudnn_handle, bn_mode, &alphaDataDiff, &betaDataDiff,
                                                         &alphaParamDiff, &betaParamDiff, xDesc, input->ptr,
                                                         yDesc, delta->ptr, xDesc, parent[0]->delta->ptr,
                                                         bnScaleBiasMeanVarDesc,bn_g->ptr, gbn_g->ptr, gbn_b->ptr,
                                                         epsilon, bn_mean->ptr, bn_var->ptr);
    if(nnn != CUDNN_STATUS_SUCCESS) std::cout<<"Error bwd BN  "<< cudnnGetErrorString(nnn) <<std::endl;
#endif

}



Layer *LBatchNorm::share(int c, int bs, vector<Layer *> p) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;
    n->trainable = trainable;


    //share params and gradients
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    for (int i = 0; i < n->gradients.size(); i++) delete n->gradients[i];
    n->gradients.clear();

    if (affine) {
        n->bn_g=bn_g;
        n->bn_b=bn_b;
        n->params.push_back(bn_g);
        n->params.push_back(bn_b);

        n->gbn_g=gbn_g;
        n->gbn_b=gbn_b;
        n->gradients.push_back(gbn_g);
        n->gradients.push_back(gbn_b);
    }
    n->mean=mean;
    n->variance=variance;
    n->params.push_back(mean);
    n->params.push_back(variance);

    return n;
}

Layer *LBatchNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine,  name, todev,mem_level);
    n->orig = this;
    n->trainable = trainable;

    return n;
}


string LBatchNorm::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
