
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

#include "layer_conv.h"


using namespace std;

int LUpSampling::total_layers = 0;

LUpSampling::LUpSampling(Layer *parent, const vector<int> &size, string interpolation, string name, int dev) : LinLayer(name, dev) {
    // TODO: Implement
    this->size = size;
    this->interpolation = interpolation;
    if(name.empty()) this->name = "upsampling" + to_string(++total_layers);
}