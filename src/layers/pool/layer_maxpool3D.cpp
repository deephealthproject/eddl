/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/pool/layer_pool.h"


using namespace std;


// ---- MAXPOOL3D ----

// constructors and clones
LMaxPool3D::LMaxPool3D(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem) : LMaxPool3D(parent, new PoolDescriptor3D(pool_size, strides, padding, mem), name, dev, mem) {}

LMaxPool3D::LMaxPool3D(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, const string& name, int dev, int mem) : LMaxPool3D(parent, new PoolDescriptor3D(pool_size, strides, padding, mem), name, dev, mem) {}

LMaxPool3D::LMaxPool3D(Layer *parent, PoolDescriptor3D *D, const string& name, int dev, int mem) : LPool3D(parent, D, name, dev, mem) {
    if(name.empty()) this->name = "maxpool3d" + to_string(++total_layers);

    // Params
    D->indX = new Tensor(D->O->shape, dev);  // Is this needed here?
    D->indY = new Tensor(D->O->shape, dev);

}


void LMaxPool3D::resize(int batch){
  LPool3D::resize(batch);

  delete pd->indX; pd->indX = new Tensor(pd->O->shape, dev);
  delete pd->indY; pd->indY = new Tensor(pd->O->shape, dev);
}

void LMaxPool3D::forward() {
    tensorNN::MPool3D(this->pd);
}

void LMaxPool3D::backward() {
    tensorNN::MPool3D_back(this->pd);
}

Layer *LMaxPool3D::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LMaxPool3D(p[0], this->pd, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LMaxPool3D::clone(int c, int bs, vector<Layer *> p, int todev) {

    auto *n = new LMaxPool3D(p[0], new PoolDescriptor3D(pd->ksize, pd->stride, pd->pad, pd->mem_level),  "share_"+to_string(c)+this->name, todev, this->mem_level);

    n->orig = this;
    return n;
}

string LMaxPool3D::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
