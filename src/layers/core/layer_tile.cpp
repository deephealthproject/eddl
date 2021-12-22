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

int LTile::total_layers = 0;

LTile::LTile(Layer *parent, const vector<int>& repeats, string name, int dev, int mem) : LinLayer(name, dev, mem, "tile") {
    if(name.empty()) this->name = "tile" + to_string(++total_layers);

    // Check dimensions
    if(parent->output->ndim != repeats.size()){
        msg("The number of dimensions of the output layer must match the size of 'repeats'", "LTile::LTile");
    }

    // Dimensions must be positive
    for(int i=0; i<repeats.size(); i++){
        if(repeats[i] < 1){
            msg("All repetitions must be greater or equal than 1", "LTile::LTile");
        }
    }

    // Set input
    input = parent->output;

    // Get input shape, but making sure the batch dimension is 1
    vector<int> input_shape_single_batch(input->shape.begin()+1, input->shape.end());
    input_shape_single_batch.insert(input_shape_single_batch.begin(), 1);

    // Build descriptor (batch must equal to "1")
    this->td = new TileDescriptor(repeats, dev);
    this->td->build(input_shape_single_batch);

    // Set output tensors
    output=new Tensor(this->td->oshape, dev);

    // Set parent
    parent->addchild(this);
    addparent(parent);
}

LTile::~LTile(){
    delete td;
}

void LTile::forward(){
    tensorNN::select(this->input, this->output, this->td);
}

void LTile::backward(){
    tensorNN::select_back(this->delta, this->parent[0]->delta, this->td);
}

Layer *LTile::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LTile(p[0], this->td->vrepeats, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LTile::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LTile(p[0], this->td->vrepeats, this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LTile::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
