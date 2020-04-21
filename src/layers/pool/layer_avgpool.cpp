/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/pool/layer_pool.h"


using namespace std;


// ---- MAXPOOL2D ----
// constructors and clones

// constructors and clones
LAveragePool::LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem) : LAveragePool(parent, new PoolDescriptor(pool_size, strides, padding, mem), name, dev, mem) {}

LAveragePool::LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, string name, int dev, int mem) : LAveragePool(parent, new PoolDescriptor(pool_size, strides, padding, mem), name, dev, mem) {}

LAveragePool::LAveragePool(Layer *parent, PoolDescriptor *D, const string& name, int dev, int mem) : LPool(parent, D, name, dev, mem) {
    if(name.empty()) this->name = "avgpool" + to_string(++total_layers);
}


void LAveragePool::resize(int batch){
    LPool::resize(batch);
}

void LAveragePool::forward() {
    AvgPool2D(this->pd);
}

void LAveragePool::backward() {
    AvgPool2D_back(this->pd);
}

Layer *LAveragePool::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LAveragePool(p[0], this->pd, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LAveragePool::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LMaxPool(p[0], new PoolDescriptor(pd->ksize, pd->stride, pd->padding, pd->mem_level), "clone_" + to_string(todev) + this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}

string LAveragePool::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
