// The MIT License (MIT)
//
// Copyright (c) 2019 PRHLT Research Group. Inc. http://www.prhlt.upv.es
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
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "loss.h"

using namespace std;


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