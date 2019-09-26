
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

#include "layer_pool.h"


using namespace std;


int LPool::total_layers = 0;

LPool::LPool(Layer *parent, PoolDescriptor *D, string name, int dev) : LinLayer(name, dev) {
    if (parent->output->ndim != 4) msg("LPool only works over 4D tensors", "LPool::LPool");
    if(name.empty()) this->name = "pool" + to_string(++total_layers);

    pd = D;

    input = parent->output;
    pd->build(input);

    output = pd->O;
    delta = pd->D;
    pd->ID = parent->delta;

    parent->addchild(this);
    addparent(parent);

}

void LPool::resize(int batch){
  //cout<<"Resize "<<name<<"\n";

  input = parent[0]->output;
  pd->resize(input);

  output = pd->O;
  delta = pd->D;
  pd->ID = parent[0]->delta;

}
