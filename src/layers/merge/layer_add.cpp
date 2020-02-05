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

#include "layer_merge.h"


using namespace std;


int LAdd::total_layers = 0;



LAdd::LAdd(vector<Layer *> parent, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if (parent.size() == 0) msg("Error: LAdd layer with empty list");

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::eqsize(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LAdd layers with different tensor shape");
            }

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->getShape(), dev);
    if (!mem_level) delta = new Tensor(parent[0]->output->getShape(), dev);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LAdd::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LAdd::forward() {
    output->fill_(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);

}

void LAdd::backward() {
    for (int i = 0; i < parent.size(); ++i) {
      if (parent[i]->mem_level)  parent[i]->mem_delta();
        Tensor::inc(delta, parent[i]->delta);
      }

    if (mem_level)  free_delta();
}

void LAdd::resize(int batch){
  Layer::resize(batch);
}

Layer *LAdd::share(int c, int bs, vector<Layer *> p) {
    LAdd *n = new LAdd(p, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}


Layer *LAdd::clone(int c, int bs, vector<Layer *> p, int todev) {
    LAdd *n = new LAdd(p, "share_" + to_string(c) + name, todev,mem_level);
    n->orig = this;

    return n;
}





///////////////
