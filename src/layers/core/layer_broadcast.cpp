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

#include "eddl/layers/core/layer_core.h"


using namespace std;

int LBroadcast::total_layers = 0;

LBroadcast::LBroadcast(Layer *parent1, Layer *parent2, string name, int dev, int mem) : LinLayer(name, dev, mem, "broadcast") {
    // Important:
    // There are two parents, but one of them is "x" and the other provides the constant shape to which broadcast
    // To give more flexibility, the "x" is going to be the parent from which we are going to broadcast.
    // Example:
    //      - f(P1(3), P2(4,2,3,5)) => P1 is x.  (P2 has no delta)
    //      - f(P1(4,2,3,5), P2(3)) => P2 is x.  (P1 has no delta)
    if(name.empty()) this->name = "broadcast" + to_string(++total_layers);

    // Get shapes
    vector<int> shape1 = parent1->output->shape;
    vector<int> shape2 = parent2->output->shape;
    this->shapes_swapped = false;

    // Remove batch dimension (some checks are done at shape[0]
    shape1 = vector<int>(shape1.begin()+1, shape1.end());
    shape2 = vector<int>(shape2.begin()+1, shape2.end());

    // Determine which parent is the smaller one
    if(shape1.size()<=shape2.size()) {
        this->input = parent1->output;  // Input must be the output of the smaller
    }else{
        this->input = parent2->output;  // Input must be the output of the smaller
        shape1.swap(shape2);
        this->shapes_swapped = true;
    }

    // Get shape to broadcast (normalized)
    vector<int> broadcast_from = getBroadcastShape(shape1, shape2);
    if(broadcast_from.empty()){
        msg("The dimensions of both tensors must be equal or compatible (i.e (3)*(1,3), (3)*(6,2,5,3,5), (5, 3)*(5, 3),...)", "Tensor::broadcast");
    }

    // Get repetitions to perform a given broadcast
    vector<int> tile_repetitions = getTilesRepetitions(broadcast_from, shape2);
    if(tile_repetitions.empty()){
        msg("These tensors cannot be broadcasted. Two dimensions are compatible when: 1) they are equal, or 2) one of them is 1", "Tensor::broadcast");
    }

    // Build descriptor (without batch)
    this->td = new TileDescriptor(tile_repetitions, dev);
    td->build(broadcast_from);

    // Set output tensor (add batch)
    vector<int> oshape(this->td->oshape); oshape.insert(oshape.begin(), 1);  // Add batch with dimension "1" (not needed, but this is for consistency)
    output = new Tensor(oshape, dev);

    // Set parents
    if(!this->shapes_swapped){  // parent1 smaller than parent2
        parent1->addchild(this);
        addparent(parent1);
        p1=parent1; p2=parent2;
    }else{   // parent2 smaller than parent1
        parent2->addchild(this);
        addparent(parent2);
        p1=parent2; p2=parent1;
    }
}

LBroadcast::~LBroadcast(){
    this->p1 = nullptr;  // There are simply to store a reference
    this->p2 = nullptr;  // There are simply to store a reference
    delete td;
}

void LBroadcast::forward(){
    if(!this->shapes_swapped){
        tensorNN::select(this->p1->input, this->output, this->td);
    }else{
        tensorNN::select(this->p2->input, this->output, this->td);
    }
}

void LBroadcast::backward(){
    if(!this->shapes_swapped){
        tensorNN::select_back(this->delta, this->p1->delta, this->td);
    }else{
        tensorNN::select_back(this->delta, this->p2->delta, this->td);
    }
}

Layer *LBroadcast::share(int c, int bs, vector<Layer *> p){
    auto *n = new LBroadcast(p[0], this->p2, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LBroadcast::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LBroadcast(p[0], this->p2, this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LBroadcast::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
