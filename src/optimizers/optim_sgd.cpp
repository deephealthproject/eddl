/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/optimizers/optim.h"

using namespace std;


SGD::SGD(float lr, float momentum, float weight_decay, bool nesterov) : Optimizer() {
    this->lr = lr;
    this->mu = momentum;
    this->weight_decay = weight_decay;
    this->nesterov = nesterov;

}

SGD::~SGD() {
    for(int i=0; i<mT.size(); i++){ delete mT[i]; }
}

void SGD::change(vector<float> p) {
    if (p.size()>0) lr = p[0];
    if (p.size()>1) mu = p[1];
}

Optimizer *SGD::clone() {
    SGD *n=new SGD(lr, mu, weight_decay, nesterov);
    n->clip_val=clip_val;

    return n;
}

Optimizer *SGD::share() {
    SGD *n=new SGD(lr, mu, weight_decay, nesterov);
    n->orig=this;
    n->isshared=true;
    n->clip_val=clip_val;
    return n;
}

void SGD::setlayers(vlayer l) {
    layers = l;

    if (isshared) return;

    // create momemtum tensors
    for (int i = 0; i < layers.size(); i++)
      for (int j = 0; j < layers[i]->get_trainable_params_count(); j++) {
          mT.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
          mT.back()->fill_(0.0);
    }


}

void SGD::applygrads(int batch) {
    if (isshared) {
      orig->applygrads(batch);
    }
    else {
      clip();
      int p = 0;
      for (int i = 0; i < layers.size(); i++) {
        if (layers[i]->trainable) {
          for (int j = 0; j < layers[i]->get_trainable_params_count(); j++, p++) {
            Tensor::add(lr , layers[i]->gradients[j], mu, mT[p], mT[p], 0);
            Tensor::add(1.0, layers[i]->params[j], -1.0, mT[p], layers[i]->params[j], 0);

            // Distributed training: Accumulation of gradients
            if (layers[i]->acc_gradients.size() > 0) 
              Tensor::add(1.0, layers[i]->acc_gradients[j], -1.0, mT[p], layers[i]->acc_gradients[j], 0);
          }
        }
        else p+=layers[i]->get_trainable_params_count();
      }
    }
}
