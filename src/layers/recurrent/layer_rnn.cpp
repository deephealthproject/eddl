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

#include "eddl/layers/recurrent/layer_recurrent.h"


using namespace std;

int LRNN::total_layers = 0;

LRNN::LRNN(vector<Layer *> parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name, int dev, int mem) : MLayer(name, dev, mem) {

    this->units = units;
    int ndim=units;
    this->num_layers = num_layers;
    this->use_bias = use_bias;
    this->dropout = dropout;
    this->bidirectional = bidirectional;
    isrecurrent=true;

    // TODO: Implement

    if (parent[0]->output->ndim != 2) msg("LRNN only works over 2D tensors", "LRNN");

    if(name.empty()) this->name = "RNN" + to_string(++total_layers);

    input = parent[0]->output;
    output = new Tensor(vector<int>{input->shape[0], ndim}, dev);

    // From parent layer
    Wx = new Tensor(vector<int>{input->shape[1], ndim}, dev);
    if (use_bias) biasx = new Tensor(vector<int>{ndim}, dev);
    params.push_back(Wx);
    if (use_bias) params.push_back(biasx);

    gWx = new Tensor(vector<int>{input->shape[1], ndim}, dev);
    if (use_bias) gbiasx = new Tensor(vector<int>{ndim}, dev);
    gradients.push_back(gWx);
    if (use_bias) gradients.push_back(gbiasx);

    // From t-1 RNN
    Wy = new Tensor(vector<int>{ndim, ndim}, dev);
    if (use_bias) biasy = new Tensor(vector<int>{ndim}, dev);
    params.push_back(Wy);
    if (use_bias) params.push_back(biasy);

    gWy = new Tensor(vector<int>{ndim, ndim}, dev);
    if (use_bias) gbiasy = new Tensor(vector<int>{ndim}, dev);
    gradients.push_back(gWy);
    if (use_bias) gradients.push_back(gbiasy);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual
void LRNN::forward() {
    // TODO: Implement
}

void LRNN::backward() {
    // TODO: Implement
}


Layer *LRNN::share(int c, int bs, vector<Layer *> p) {
    LRNN *n = new LRNN(p, units, num_layers, use_bias, dropout, bidirectional, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LRNN::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRNN *n = new LRNN(p, units, num_layers, use_bias, dropout, bidirectional, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LRNN::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=Orange,shape=polygon]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=Orange,shape=polygon]";

    return s;
}
