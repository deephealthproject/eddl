
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

#include "metric.h"

using namespace std;

MMeanSquaredError::MMeanSquaredError() : Metric("mean_squared_error"){}

float MMeanSquaredError::value(Tensor *T, Tensor *Y) {
    float f;
    // batch error: sum((T-Y)^2)
    Tensor *aux1 = new Tensor(T->getShape(), T->device);
    
    Tensor::sum(1.0, T, -1.0, Y, aux1, 0);
    Tensor::el_mult(aux1, aux1, aux1, 0);
    f = aux1->total_sum();


    delete aux1;
    return f;
}
