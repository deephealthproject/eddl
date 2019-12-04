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

#include "optim.h"

using namespace std;


RMSProp::RMSProp(float lr, float rho, float epsilon, float weight_decay) : Optimizer() {
    this->lr = lr;
    this->rho = rho;
    this->epsilon = epsilon;
    this->weight_decay = weight_decay;

}

RMSProp::~RMSProp() {
  gT1.clear();
  gT.clear();
}

void RMSProp::change(vector<float> &p) {
  if (p.size()>0) lr = p[0];
  if (p.size()>1) rho = p[1];
  cout<<"Optimizer RMSProp set new lr="<<lr<<" rho="<<rho<<"\n";
}

Optimizer *RMSProp::clone() {
    return new RMSProp(lr, rho, epsilon, weight_decay);
}

void RMSProp::setlayers(vlayer l) {
    layers = l;

    // create momemtum tensors
    for (int i = 0; i < layers.size(); i++)
        for (int j = 0; j < layers[i]->gradients.size(); j++) {
            gT1.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
            gT1.back()->fill_(0.0);
            gT.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
            gT.back()->fill_(0.0);
        }

}

void RMSProp::applygrads(int batch) {

    int p = 0;
    for (int i = 0; i < layers.size(); i++) {
        for (int j = 0; j < layers[i]->gradients.size(); j++, p++) {
            Tensor::copy(layers[i]->gradients[j],gT[p]);
            gT[p]->sqr_();
            gT[p]->mult_(1.0-rho);

            gT1[p]->sqr_();
            gT1[p]->mult_(rho);
            Tensor::add(1.0,gT1[p],1.0,gT[p],gT[p],0);

            gT[p]->add_(epsilon);
            gT[p]->sqrt_();
            Tensor::el_div(layers[i]->gradients[j],gT[p],gT[p],0);

            Tensor::copy(layers[i]->gradients[j],gT1[p]);

            Tensor::add(-lr, gT[p],1.0,layers[i]->params[j], layers[i]->params[j], 0);
        }
    }


}
