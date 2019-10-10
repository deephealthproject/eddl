/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
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


SGD::SGD(float lr, float momentum, float weight_decay, bool nesterov) : Optimizer() {
    this->lr = lr;
    this->mu = momentum;
    this->weight_decay = weight_decay;
    this->nesterov = nesterov;

}

void SGD::change(vector<float> &p) {
    lr = p[0];
    mu = p[1];
}

Optimizer *SGD::clone() {
    return new SGD(lr, mu, weight_decay, nesterov);
}

void SGD::setlayers(vlayer l) {
    layers = l;

    // create momemtum tensors
    for (int i = 0; i < layers.size(); i++)
        for (int j = 0; j < layers[i]->gradients.size(); j++) {
            mT.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
            mT.back()->set(0.0);
        }

}

void SGD::applygrads(int batch) {

    int p = 0;

    for (int i = 0; i < layers.size(); i++) {
        //printf("=====\n%s\n=====\n",layers[i]->name.c_str());

        for (int j = 0; j < layers[i]->gradients.size(); j++, p++) {
            //layers[i]->params[j]->info();
            Tensor::add(lr / batch, layers[i]->gradients[j], mu, mT[p], mT[p], 0);
            Tensor::add(1.0, layers[i]->params[j], 1.0, mT[p], layers[i]->params[j], 0);
        }
        //printf("\n\n");
    }
    //getchar();

}
