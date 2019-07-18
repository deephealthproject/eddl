
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

#include "layer_generators.h"


using namespace std;

GeneratorLayer::GeneratorLayer(string name, int dev) : Layer(name, dev) {
    binary=0;
}


void GeneratorLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void GeneratorLayer::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}
