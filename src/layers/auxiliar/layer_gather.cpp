/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/auxiliar/layer_auxiliar.h"


using namespace std;

int LGather::total_layers = 0;

LGather::LGather(Layer *parent, int axis, Tensor* indices, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "gather" + to_string(++total_layers);

    this->axis = axis;
    this->indices = indices;

    // TODO dummu code
    // Build descriptor
    vector<int> shape_no_batch(input->shape.begin()+1, input->shape.end());
    sd = new GatherDescriptor({}, dev);
    sd->build(shape_no_batch);  // Ignore batch

    // Set flow tensors
    vector<int> oshape(sd->oshape);
    oshape.insert(oshape.begin() + 0, 1);

    input = parent->output;
    output=new Tensor(input->shape, dev);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LGather::resize(int batch){
    output->resize(batch);
}


void LGather::forward() {
    Tensor::gather(this->input, this->output, this->sd);
}

void LGather::backward() {
    msg("NotImplementedError", "LGather::backward");
}


Layer *LGather::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LGather(p[0], this->axis, this->indices, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LGather::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LGather(p[0], this->axis, this->indices, name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LGather::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
