/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "layers/pool/layer_pool.h"


using namespace std;


// ---- MAXPOOL2D ----
// constructors and clones

// constructors and clones
LMaxPool::LMaxPool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem) : LMaxPool(parent, new PoolDescriptor(pool_size, strides, padding, mem), name, dev, mem) {}

LMaxPool::LMaxPool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, const string& name, int dev, int mem) : LMaxPool(parent, new PoolDescriptor(pool_size, strides, padding, mem), name, dev, mem) {}

LMaxPool::LMaxPool(Layer *parent, PoolDescriptor *D, const string& name, int dev, int mem) : LPool(parent, D, name, dev, mem) {
    if(name.empty()) this->name = "maxpool" + to_string(++total_layers);

    // Params
    D->indX = new Tensor(D->O->shape, dev);
    D->indY = new Tensor(D->O->shape, dev);
}


void LMaxPool::resize(int batch){
  LPool::resize(batch);

  delete pd->indX;
  delete pd->indY;

  pd->indX = new Tensor(pd->O->shape, dev);
  pd->indY = new Tensor(pd->O->shape, dev);

}

void LMaxPool::forward() {
    MPool2D(this->pd);
}

void LMaxPool::backward() {
    MPool2D_back(this->pd);
}

Layer *LMaxPool::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LMaxPool(p[0], this->pd, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LMaxPool::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LMaxPool(p[0], this->pd, "clone_" + to_string(todev) + this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}

string LMaxPool::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
