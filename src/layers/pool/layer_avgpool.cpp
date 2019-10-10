/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


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
