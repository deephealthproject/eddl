
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


// ---- AVERAGE POOL ----
LAveragePool::LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, string padding, string name, int dev) : LAveragePool(parent, new PoolDescriptor(pool_size, strides, padding), name, dev) {}
LAveragePool::LAveragePool(Layer *parent, PoolDescriptor *D, string name, int dev) : LPool(parent, D, name, dev) {
    // TODO: Implement
}
