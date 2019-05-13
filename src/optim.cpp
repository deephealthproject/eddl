// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "optim.h"

using namespace std;

optim::optim() {

}


////// SGD //////
sgd::sgd(const initializer_list<float> &p) : optim() {
    vector<float> v = vector<float>(p.begin(), p.end());
    lr = v[0];
    mu = v[1];
}

void sgd::change(const initializer_list<float> &p) {
    vector<float> v = vector<float>(p.begin(), p.end());
    lr = v[0];
    mu = v[1];
}

optim *sgd::clone() {
    return new sgd({lr, mu});
}


void sgd::setlayers(vlayer l) {
    layers = l;

    // create momemtum tensors
    for (int i = 0; i < layers.size(); i++)
        for (int j = 0; j < layers[i]->gradients.size(); j++) {
            mT.push_back(new Tensor(layers[i]->gradients[j]->getShape(), layers[i]->dev));
            mT.back()->set(0.0);
        }

}


void sgd::applygrads(int batch) {

    int p = 0;

    for (int i = 0; i < layers.size(); i++) {
        for (int j = 0; j < layers[i]->gradients.size(); j++, p++) {
            Tensor::sum(lr / batch, layers[i]->gradients[j], mu, mT[p], mT[p], 0);
            Tensor::sum(1.0, layers[i]->params[j], 1.0, mT[p], layers[i]->params[j], 0);
        }
    }
    //getchar();

}


///////////////////////////////////////////

//////
