/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/optimizers/optim.h"

using namespace std;


RMSProp::RMSProp(float lr, float rho, float epsilon, float weight_decay) : Optimizer() {
    this->lr = lr;
    this->rho = rho;
    this->epsilon = epsilon;
    this->weight_decay = weight_decay;

}

RMSProp::~RMSProp() {
    for(int i=0; i<gT.size(); i++){ delete gT[i]; gT[i] = nullptr; }
    for(int i=0; i<gT1.size(); i++){ delete gT1[i]; gT1[i] = nullptr;}
}

void RMSProp::change(vector<float> p) {
  if (p.size()>0) lr = p[0];
  if (p.size()>1) rho = p[1];
  cout<<"Optimizer RMSProp set new lr="<<lr<<" rho="<<rho<<"\n";
}

Optimizer *RMSProp::clone() {
    RMSProp *n=new RMSProp(lr, rho, epsilon, weight_decay);
    n->clip_val=clip_val;

    return n;
}

Optimizer *RMSProp::share() {
    RMSProp *n=new RMSProp(lr, rho, epsilon, weight_decay);
    n->orig=this;
    n->isshared=true;
    n->clip_val=clip_val;
    return n;
}
void RMSProp::setlayers(vlayer l) {
    layers = l;

    if (isshared) return;

    // create momemtum tensors
    for (int i = 0; i < layers.size(); i++){
        for (int j = 0; j < layers[i]->get_trainable_params_count(); j++) {
            gT1.emplace_back(Tensor::zeros_like(layers[i]->gradients[j]));
            gT.emplace_back(Tensor::zeros_like(layers[i]->gradients[j]));
        }
    }
}

void RMSProp::applygrads(int batch) {
  if (isshared) {
    orig->applygrads(batch);
  }
  else {

    clip();

    int p = 0;
    for (int i = 0; i < layers.size(); i++)
      if (layers[i]->trainable) {
        for (int j = 0; j < layers[i]->get_trainable_params_count(); j++, p++) {
            Tensor::copy(layers[i]->gradients[j],gT[p]);
            gT[p]->sqr_();
            gT[p]->mult_(1.0f-rho);

            gT1[p]->sqr_();
            gT1[p]->mult_(rho);
            Tensor::add(1.0,gT1[p],1.0,gT[p],gT[p],0);

            gT[p]->add_(epsilon);
            gT[p]->sqrt_();
            Tensor::el_div(layers[i]->gradients[j],gT[p],gT[p],0);

            Tensor::copy(layers[i]->gradients[j],gT1[p]);

            Tensor::add(-lr, gT[p],1.0,layers[i]->params[j], layers[i]->params[j], 0);

            // Distributed training: Accumulation of gradients
            if (layers[i]->acc_gradients.size() > 0) 
              Tensor::add(-lr, gT[p],1.0,layers[i]->acc_gradients[j], layers[i]->acc_gradients[j], 0);
        }
    }
    else p+=layers[i]->get_trainable_params_count();
  }

}
