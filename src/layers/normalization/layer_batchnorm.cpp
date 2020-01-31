/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
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

int LBatchNorm::total_layers = 0;


LBatchNorm::LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev, int mem) : LinLayer(name, dev) {

    input=parent->output;
    mem_level=mem;

    if (input->ndim == 2) {axis.push_back(0);shape.push_back(input->shape[1]);}
    else if (input->ndim == 4) {axis.push_back(0);axis.push_back(2);axis.push_back(3);shape.push_back(input->shape[1]);}
    else {
      input->info();
      msg("LBatchNorm only works over 1D (Dense) or 2D (Conv) tensors","LBatchNorm");
    }


    if(name.empty()) this->name = "batchnorm" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    output=new Tensor(input->getShape(),dev);
    if (mem_level<2) delta=new Tensor(input->getShape(),dev);

    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);


    MD=new MapReduceDescriptor(input,axis);

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
void LBatchNorm::resize(int batch){
  if (batch!=output->shape[0]) {
    output->resize(batch);
    if (mem_level<2) delta->resize(batch);
    if (target!=nullptr) target->resize(batch);
    delete MD;
    MD=new MapReduceDescriptor(input,axis);
  }


}

void LBatchNorm::save(std::ofstream &ofs, string format){
  // Save momentum TODO
  if (momentum!=0) {
    mean->savefs(ofs, format);
    variance->savefs(ofs, format);
  }
}

void LBatchNorm::load(std::ifstream &ifs, string format){
    // load momentum TODO
    if (momentum!=0) {
      Tensor *t=mean->loadfs(ifs, format);
      Tensor::copy(t,mean);
      delete t;
      t=variance->loadfs(ifs, format);
      Tensor::copy(t,variance);
      delete t;

    }

}

void LBatchNorm::copy(Layer *l2)
{
  Tensor::copy(mean,((LBatchNorm*)l2)->mean);
  Tensor::copy(variance,((LBatchNorm*)l2)->variance);
}


void LBatchNorm::forward() {
  BN_forward(input,output,MD,bn_mean,bn_var,mean,variance,momentum,epsilon,mode==TRMODE);
}

void LBatchNorm::backward()
{
  if (parent[0]->mem_level==2) parent[0]->mem_delta();
  BN_backward(input,delta, parent[0]->delta,MD,bn_mean,bn_var,mean,variance,epsilon);
  if (mem_level==2) free_delta();

}



Layer *LBatchNorm::share(int c, int bs, vector<Layer *> p) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LBatchNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev,mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LBatchNorm::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
