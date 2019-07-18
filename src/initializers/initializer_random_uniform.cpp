
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

#include "initializer.h"

using namespace std;


IRandomUniform::IRandomUniform(float minval, float maxval, int seed) : Initializer("random_uniform") {
    // Todo: Implement
    this->minval = minval;
    this->maxval = maxval;
    this->seed = seed;

}
float IRandomUniform::set_weights(Tensor* T) { return 0; }
