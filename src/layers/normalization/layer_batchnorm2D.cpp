/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_normalization.h"
#include "../reductions/layer_reductions.h"
#include "../operators/layer_operators.h"

using namespace std;

int LBatchNorm2D::total_layers = 0;


LBatchNorm2D::LBatchNorm2D(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev) : LinLayer(name, dev) {


    input=parent->output;

    if (input->ndim == 4) {axis.push_back(0);axis.push_back(2);axis.push_back(3);shape.push_back(input->shape[1]);}
    else msg("LBatchNorm2D only works over 2D (Conv) tensors","LBatchNorm2D");


    if(name.empty()) this->name = "batchnorm2D" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    output=new Tensor(input->getShape(),dev);
    delta=new Tensor(input->getShape(),dev);
    bn_E=new Tensor(input->getShape(),dev);

    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);
    sd=new Tensor(shape,dev);


    redmap=Tensor::get_reduction_map(input,axis);

    if (momentum!=0.0) {
        mean=new Tensor(shape,dev);
        mean->fill_(0.0);

        variance=new Tensor(shape,dev);
        variance->fill_(1.0);
    }

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LBatchNorm2D::resize(int batch){
  if (batch!=output->shape[0]) {
    output->resize(batch);
    delta->resize(batch);
    bn_E->resize(batch);
    if (target!=nullptr) target->resize(batch);
    free(redmap);
    redmap=Tensor::get_reduction_map(input,axis);
  }


}

void LBatchNorm2D::forward() {


  if (mode == TRMODE) {

    Tensor::reduce_mean(input,bn_mean, axis,redmap);

    Tensor::reduce_variance(input,bn_var, axis,redmap);

    if (momentum!=0.0) {
      Tensor::add(momentum, mean, (1.0-momentum), bn_mean,mean,0);
      Tensor::add(momentum, variance, (1.0-momentum), bn_var,variance,0);
    }

    Tensor::copy(bn_var,sd);

    sd->add_(epsilon);
    sd->sqrt_();

    Tensor::copy(input,output);

    Tensor::reduce_diff(output,bn_mean,axis,redmap);

    Tensor::reduce_div(output,sd,axis,redmap);

  }
  else { // testmode
    /*
    Tensor::reduced_sum(1,E,-1.0/bnc,bn_gmean,bn_E,0,1);
    bn_var->copy(bn_gvar);
    bn_var->div(bnc);
    bn_var->add(eps);
    bn_var->sqr();

    Tensor::reduced_div(bn_E,bn_var,bn_E,0,1);

    Tensor::reduced_mult(bn_E,bn_g,BNE,0,1);
    Tensor::reduced_sum(1,BNE,1,bn_b,BNE,0,1);
    */
  }

}

void LBatchNorm2D::backward()
{
/*

Tensor *A=new Tensor(Delta->a,Delta->b,Delta->c,Delta->d);
Tensor *Tvar32=bn_var->Clone();
Tensor *Tsqvar=bn_var->Clone();
float eps=0.0001;

// No affine

//4 Var
Tsqvar->add(eps);
Tsqvar->sqr();

Tvar32->add(eps);

Tensor::el_mult(Tvar32,0,Tsqvar,0,Tvar32,0);
Tensor::reduced_sum(-0.5,E,0.5,bn_mean,A,0,1);
Tensor::reduced_div(A,Tvar32,A,0,1);
Tensor::el_mult(A,0,gbn_E,0,A,0);
Tensor::reduceTosum(A,gbn_var,1);

//5 Mean
Tensor::reduced_div(gbn_E,Tsqvar,A,0,1);
A->mul(-1);
Tensor::reduceTosum(A,gbn_mean,1);

//6 Delta
int m=batch*outr*outc;
Tensor::reduced_div(gbn_E,Tsqvar,Delta,0,1);
Tensor::reduced_sum(2.0/m,E,-2.0/m,bn_mean,A,0,1);
Tensor::reduced_mult(A,gbn_var,A,0,1);
Tensor::reduced_sum(1,A,1.0/m,gbn_mean,Delta,1,1);


delete A;
delete Tvar32;
delete Tsqvar;
*/
}



Layer *LBatchNorm2D::share(int c, int bs, vector<Layer *> p) {
    LBatchNorm2D *n = new LBatchNorm2D(p[0], momentum, epsilon, affine, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LBatchNorm2D::clone(int c, int bs, vector<Layer *> p, int todev) {
    LBatchNorm2D *n = new LBatchNorm2D(p[0], momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LBatchNorm2D::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
