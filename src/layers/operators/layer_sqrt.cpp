
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

#include "layer_operators.h"


using namespace std;

int LSqrt::total_layers = 0;

/**
  @brief Computes square root of a Layer element-wise

  @param l a Layer.
  @param name a name for the operation (predefined as 'sqrt+TotalSqrtLayers')
  @param dev which computing service utilize

  @returns the result of the square root operation over l

  */
LSqrt::LSqrt(Layer *l, string name, int dev): OperatorLayer(name, dev) {
    if(name.empty()) this->name = "sqrt" + to_string(++total_layers);
    //TODO: Implement
}

void LSqrt::forward(){
    //TODO: Implement
}

void LSqrt::backward(){
    //TODO: Implement
}

Layer *LSqrt::share(int c, int bs, vector<Layer *> p) {

    return nullptr;
}

Layer *LSqrt::clone(int c, int bs, vector<Layer *> p, int todev) {

    return nullptr;
}
