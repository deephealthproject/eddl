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

#include "loss.h"

using namespace std;

Loss::Loss(string name) {
    this->name = name;
}
void Loss::delta(Tensor *T, Tensor *Y, Tensor *D) {}
float Loss::value(Tensor *T, Tensor *Y) {return 0;}


LMeanSquaredError::LMeanSquaredError() : Loss("mean_squared_error"){}
void LMeanSquaredError::delta(Tensor *T, Tensor *Y, Tensor *D) {
    //delta: (T-Y)
    Tensor::sum(1.0, T, -1.0, Y, D, 0);
}
float LMeanSquaredError::value(Tensor *T, Tensor *Y) {
    float f;
    // batch error: sum((T-Y)^2)
    Tensor *aux = new Tensor(T->getShape(), T->device);
    Tensor::sum(1.0, T, -1.0, Y, aux, 0);
    Tensor::el_mult(aux, aux, aux, 0);
    f = aux->total_sum();
    delete aux;
    return f;
}


LCrossEntropy::LCrossEntropy() : Loss("cross_entropy"){}
void LCrossEntropy::delta(Tensor *T, Tensor *Y, Tensor *D) {
    // delta: t/y - (1-t)/(1-y)
    Tensor *aux1 = new Tensor(T->getShape(), T->device);
    Tensor *aux2 = new Tensor(T->getShape(), T->device);
    Tensor *one = new Tensor(T->getShape(), T->device);
    one->set(1.0);

    //  (1-t)/(1-y)
    Tensor::sum(1, one, -1, T, aux1, 0);
    Tensor::sum(1, one, -1, Y, aux2, 0);
    Tensor::el_div(aux1, aux2, aux2, 0);

    // t/y
    Tensor::el_div(T, Y, aux1, 0);

    Tensor::sum(1, aux1, -1, aux2, D, 0);

    delete aux1;
    delete aux2;
    delete one;
}
float LCrossEntropy::value(Tensor *T, Tensor *Y) {
    float f;
    Tensor *aux = new Tensor(T->getShape(), T->device);
    Tensor::cent(T, Y, aux);
    f = aux->total_sum();
    delete aux;
    return f;
}


LSoftCrossEntropy::LSoftCrossEntropy() : Loss("soft_cross_entropy"){}
void LSoftCrossEntropy::delta(Tensor *T, Tensor *Y, Tensor *D) {
    Tensor::sum(1.0, T, -1.0, Y, D, 0);
}
float LSoftCrossEntropy::value(Tensor *T, Tensor *Y) {
    float f;
    Tensor *aux = new Tensor(T->getShape(), T->device);
    Tensor::cent(T, Y, aux);
    f = aux->total_sum();
    delete aux;
    return f;
}