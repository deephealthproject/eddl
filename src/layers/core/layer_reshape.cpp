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

#include "layer_core.h"

extern ostream &operator<<(ostream &os, const vector<int> shape);


using namespace std;

int LReshape::total_layers = 0;

LReshape::LReshape(Layer *parent, vector<int> shape, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    ls = shape;

    if(name.empty()) this->name = "reshape" + to_string(++total_layers);

    input = parent->output;

    vector<int> sin = input->getShape();
    int tin = input->size;
    int t = 1, c = 0, ind = -1;

    // Check shape comp.
    for (int i = 0; i < ls.size(); i++) {
        if (ls[i] != -1) t *= ls[i];
        else {
            if (c) msg("Ambiguous reshape, more than one -1", "Reshape");
            else {
                c = 1;
                ind = i;
            }
        }
    }

    if (c == 1) {

        if (t > tin) {
            msg("Incompatible shape", "Reshape");
        } else if (tin % t) {
            msg("Incompatible shape", "Reshape");
        } else {
            ls[ind] = tin / t;
            t = tin;
        }
    } else if (t != tin) {
        msg("Incompatible shape", "Reshape");
    }

    ///////

    // sharing the pointers to data
    output = new Tensor(ls, parent->output);
    delta = new Tensor(ls, parent->delta);

    parent->addchild(this);
    addparent(parent);
}

LReshape::~LReshape()
{
    output=delta=nullptr;
}

// virtual
void LReshape::resize(int batch){
    ls[0]=batch;
    output->resize(batch, parent[0]->output);
    delta->resize(batch, parent[0]->delta);
    if (target!=nullptr) target->resize(batch);
}


void LReshape::mem_delta() {
    // Note: Tricky function when full_mem! (until future changes)
    // Reserve parent's delta AND assign it to this layer
    if (parent[0]->mem_level)  {
        parent[0]->mem_delta();

        // Problem: Delta is always created, regardless of the low_mem
        if (delta == nullptr) { // For future use!
            // When low_mem enabled, we need to build it here (the first time) as the "parent->delta" can be "nullptr"
            //delta = new Tensor(ls, parent[0]->delta);
        }else{
            // If "delta" exists, update only its "ptr"
//            delta->deleteData();
            delta->reallocate(parent[0]->delta, &ls);  //  "&ls" is needed until delta is taken out of the constructor
        }
    }
}


void LReshape::free_delta() {
    // Special case:
    // If we make "delta == nullptr", it will build another tensor in "mem_delta" instead of reallocating it
}

void LReshape::forward() {

}


void LReshape::backward() {

}


Layer *LReshape::share(int c, int bs, vector<Layer *> p) {
    vector<int> shape = ls;
    shape[0] = bs;

    auto *n = new LReshape(p[0], shape, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LReshape::clone(int c, int bs, vector<Layer *> p, int todev) {

    vector<int> shape = ls;
    shape[0] = bs;


    auto *n = new LReshape(p[0], shape, "clone_" + to_string(todev) + name, todev, mem_level);
    n->orig = this;

    return n;
}


string LReshape::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
