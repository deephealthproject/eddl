
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


Adam::Adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay, bool amsgrad) : Optimizer() {
    this->lr = lr;
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->epsilon = epsilon;
    this->weight_decay = weight_decay;
    this->amsgrad = amsgrad;

}