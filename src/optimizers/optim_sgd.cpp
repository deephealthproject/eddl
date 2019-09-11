
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

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
            Tensor::sum(lr / batch, layers[i]->gradients[j], mu, mT[p], mT[p], 0);
            Tensor::sum(1.0, layers[i]->params[j], 1.0, mT[p], layers[i]->params[j], 0);
        }
        //printf("\n\n");
    }
    //getchar();

}
