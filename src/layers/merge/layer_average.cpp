/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/merge/layer_merge.h"


using namespace std;


int LAverage::total_layers = 0;

LAverage::LAverage(vector<Layer *> parent, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if (parent.size() == 0) msg("Error: LAverage layer with empty list");

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::sameShape(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LAverage layers with different tensor shape");
            }

    if(name.empty()) this->name = "average" + to_string(++total_layers);

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->shape, dev);


    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LAverage::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LAverage::forward() {
    // TODO: Implement
    output->fill_(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);


}

void LAverage::backward() {
    // TODO: Implement
    for (int i = 0; i < parent.size(); ++i){
        Tensor::inc(delta, parent[i]->delta);
    }
}


Layer *LAverage::share(int c, int bs, vector<Layer *> p) {
    LAverage *n = new LAverage(p, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}


Layer *LAverage::clone(int c, int bs, vector<Layer *> p, int todev) {
    LAverage *n = new LAverage(p,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}





///////////////
