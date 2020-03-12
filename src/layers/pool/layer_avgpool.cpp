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


// ---- MAXPOOL2D ----
// constructors and clones

// constructors and clones
LAveragePool::LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, string padding, string name, int dev, int mem) : LAveragePool(parent, new PoolDescriptor(pool_size, strides, padding, mem), name, dev, mem) {}

LAveragePool::LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, string name, int dev, int mem) : LAveragePool(parent, new PoolDescriptor(pool_size, strides, padding, mem), name, dev, mem) {}

LAveragePool::LAveragePool(Layer *parent, PoolDescriptor *D, string name, int dev, int mem) : LPool(parent, D, name, dev, mem) {
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
    auto *n = new LAveragePool(p[0], vector<int>{pd->kr, pd->kc}, vector<int>{pd->sr, pd->sc}, pd->padding, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LAveragePool::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LAveragePool(p[0], vector<int>{pd->kr, pd->kc}, vector<int>{pd->sr, pd->sc}, pd->padding,
                               "clone_" + to_string(todev) + name, todev,mem_level);
    n->orig = this;

    return n;
}

string LAveragePool::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
