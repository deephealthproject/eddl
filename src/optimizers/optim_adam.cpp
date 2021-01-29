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

#include "eddl/optimizers/optim.h"

using namespace std;


Adam::Adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay, bool amsgrad) : Optimizer() {
    this->lr = lr;
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->epsilon = epsilon;
    this->weight_decay = weight_decay;
    this->amsgrad = amsgrad;

    t=0;
}

Adam::~Adam() {
    if (! this->isshared) {
        for(int i=0; i<mT.size(); i++){ delete mT[i]; }
        for(int i=0; i<vT.size(); i++){ delete vT[i]; }
        for(int i=0; i<mCap.size(); i++){ delete mCap[i]; }
        for(int i=0; i<vCap.size(); i++){ delete vCap[i]; }
    }
}

void Adam::change(vector<float> p) {
  if (p.size()>0) lr = p[0];
  cout<<"Optimizer Adam set new lr="<<lr<<"\n";
}

Optimizer *Adam::clone() {
    Adam *n=new Adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad);
    n->clip_val=clip_val;

    return n;
}
Optimizer *Adam::share() {
    Adam *n=new Adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad);
    n->orig=this;
    n->isshared=true;
    n->clip_val=clip_val;
    return n;
}
void Adam::setlayers(vlayer l) {
    //layers = l;
    layers.clear();
    for (auto _ : l) layers.push_back(_);

    if (isshared) return;

    // create momemtum tensors
    for (int i = 0; i < layers.size(); i++)
        for (int j = 0; j < layers[i]->get_trainable_params_count(); j++) {
            mT.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
            mT.back()->fill_(0.0);
            vT.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
            vT.back()->fill_(0.0);
            mCap.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
            mCap.back()->fill_(0.0);
            vCap.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
            vCap.back()->fill_(0.0);
        }

}

void Adam::applygrads(int batch) {
  if (isshared) {
    orig->applygrads(batch);
  }
  else {
    clip();
    int p = 0;
    t++;
    for (int i = 0; i < layers.size(); i++)
      if (layers[i]->trainable) {
        for (int j = 0; j < layers[i]->get_trainable_params_count(); j++, p++) {
            Tensor::add(beta_1,mT[p],(1-beta_1),layers[i]->gradients[j],mT[p],0);
            layers[i]->gradients[j]->sqr_();
            Tensor::add(beta_2,vT[p],(1-beta_2),layers[i]->gradients[j],vT[p],0);

            Tensor::copy(mT[p],mCap[p]);
            mCap[p]->div_(1-pow(beta_1,t));

            Tensor::copy(vT[p],vCap[p]);
            vCap[p]->div_(1-pow(beta_2,t));
            vCap[p]->add_(epsilon);
            vCap[p]->sqrt_();

            Tensor::el_div(mCap[p],vCap[p],mCap[p],0);

            Tensor::add(-lr, mCap[p],1.0,layers[i]->params[j], layers[i]->params[j], 0);

            // Distributed training: Accumulation of gradients
            if (layers[i]->acc_gradients.size() > 0)
              Tensor::add(-lr, mCap[p],1.0,layers[i]->acc_gradients[j], layers[i]->acc_gradients[j], 0);
        }
    }
    else p+=layers[i]->get_trainable_params_count();
  }

}
