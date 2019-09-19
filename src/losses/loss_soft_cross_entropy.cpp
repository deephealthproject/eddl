
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

#include "loss.h"

using namespace std;


LSoftCrossEntropy::LSoftCrossEntropy() : Loss("soft_cross_entropy"){}

void LSoftCrossEntropy::delta(Tensor *T, Tensor *Y, Tensor *D) {
    Tensor::sum(1.0, T, -1.0, Y, D, 0);
}

float LSoftCrossEntropy::value(Tensor *T, Tensor *Y) {
    float f;
    Tensor *aux1;

    aux1 = new Tensor(T->getShape(), T->device);
    Tensor::cent(T, Y, aux1);
    f = aux1->sum();

    delete aux1;

    return f;
}
