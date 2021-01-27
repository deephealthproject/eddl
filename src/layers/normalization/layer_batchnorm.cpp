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
    }
}

void cpu_batchnorm_forward(int b, int z, int rc,
        float *input, float *output, float *opa,
        float *global_mean, float *global_variance,
        float *affine_g, float *affine_b,
        float *mean, float *variance,
        bool trmode, float epsilon, float momentum)
{
    const int block_size = 256;
    int rcz = rc * z;
    if (trmode) {
        // compute mean and variance
        for (int j = 0; j < z; j++) mean[j] = variance[j] = 0.0;
        for (int k = 0; k < rcz; k += block_size)
            for (int i = 0; i < b; i++) {
                int p = k + i * rcz;
                for (int l = 0; l < block_size && k + l < rcz; l++, p++) {
                    int j = (k + l) / rc;
                    mean[j] += input[p];
                    variance[j] += input[p] * input[p];
                }
            }
        float N = b * rc;
        for (int j = 0; j < z; j++) {
            mean[j] = mean[j] / N;
            variance[j] = variance[j] / N - mean[j] * mean[j];
            // update global statistics
            if (momentum != 0.0) {
                global_mean[j] = momentum * global_mean[j] + (1.0 - momentum) * mean[j];
                global_variance[j] = momentum * global_variance[j] + (1.0 - momentum) * variance[j];
            }
            variance[j] = sqrt(variance[j] + epsilon);
        }
    } else {
        // TODO
    }
    // normalization
    for (int k = 0; k < rcz; k += block_size)
        for (int i = 0; i < b; i++) {
            int p = k + i * rcz;
            for (int l = 0; l < block_size && k + l < rcz; l++, p++) {
                int j = (k + l) / rc;
                float o = (input[p] - mean[j]) / variance[j];
                // affine transformation
                if (affine_g != NULL) {
                    opa[p] = o;
                    output[p] = o * affine_g[j] + affine_b[j];
                } else output[p] = o;
            }
        }
}

float maxerror(int size, float *a, float *b)
{
    float maxerror = 0.0;
    for (int i = 0; i < size; i++) {
        float diff = fabs(a[i] - b[i]); // / fabs(a[i]);
        if (diff > 1e-3) {
            printf("[%d] %e %e\n", i, a[i], b[i]);
        }
        if (diff > maxerror) maxerror = diff;
    }
    return maxerror;
}

// Batchnorm works over 2D Tensors
// Essentialy 4D Tensors are reshaped as 2D and
// Permute 4D tensors and set N,M values.
void LBatchNorm::forward() {
    // Input = Output = opa = {Batch,Channels,H,W} OR {Batch,Dim}
    // bn_mean = bn_var = mean = variance = bn_g = bn_b = {Channels} or {Dim}

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

    float *global_mean, *global_variance;
    if (input->isCPU()) {
        global_mean = new float[M];
        global_variance = new float[M];
        for (int j = 0; j < M; j++) {
            global_mean[j] = mean->ptr[j];
            global_variance[j] = variance->ptr[j];
        }
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

    if (input->isCPU()) { // new implementation for cpu
        float *output2 = new float[N * M];
        float *opa2 = new float[N * M];
        Tensor *opa_perm = input->clone();
        if (input->ndim == 4) tensorNN::permute_channels_first(opa, opa_perm);
        else Tensor::copy(opa, opa_perm);
        float *mean2 = new float[M];
        float *variance2 = new float[M];
        cpu_batchnorm_forward(input->shape[0], input->shape[1],
            input->ndim == 2 ? 1 : input->shape[2] * input->shape[3],
            input->ptr, output2, opa2,
            global_mean, global_variance,
            affine ? bn_g->ptr : NULL,
            affine ? bn_b->ptr : NULL,
            mean2, variance2, mode == TRMODE, epsilon, momentum);
        printf("M=%d N=%d ", M, N);
        printf("Output %e ", maxerror(N * M, output2, output->ptr));
        printf("Opa %e ", maxerror(N * M, opa2, opa_perm->ptr));
        printf("Mean %e ", maxerror(M, mean2, bn_mean->ptr));
        printf("Variance %e ", maxerror(M, variance2, bn_var->ptr));
        printf("Gmean %e ", maxerror(M, global_mean, mean->ptr));
        printf("Gvariance %e\n", maxerror(M, global_variance, variance->ptr));
        delete output2;
        delete mean2;
        delete variance2;
        delete opa2;
        delete global_mean;
        delete global_variance;
    }
}

void LBatchNorm::backward(){
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
