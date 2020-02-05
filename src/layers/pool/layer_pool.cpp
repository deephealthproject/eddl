/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
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


int LPool::total_layers = 0;

LPool::LPool(Layer *parent, PoolDescriptor *D, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 4) msg("LPool only works over 4D tensors", "LPool::LPool");
    if(name.empty()) this->name = "pool" + to_string(++total_layers);

    pd = D;

    input = parent->output;
    pd->build(input);

    output = pd->O;
    if (!mem_level) delta = pd->D;
    if (!parent->mem_level) pd->ID = parent->delta;

    parent->addchild(this);
    addparent(parent);

}

void LPool::resize(int batch){
  pd->resize(batch);
  if (target!=nullptr) target->resize(batch);
}
