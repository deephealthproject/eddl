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

#include "eddl/layers/noise/layer_noise.h"


using namespace std;

int LGaussianNoise::total_layers = 0;

LGaussianNoise::LGaussianNoise(Layer *parent, float stdev, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "gaussiannoise" + to_string(++total_layers);
    this->stdev = stdev;

    input = parent->output;
    output = new Tensor(input->shape, dev);
    noise = new Tensor(input->shape, dev);

    parent->addchild(this);
    addparent(parent);
}


LGaussianNoise::~LGaussianNoise(){
    delete noise;
    delta=nullptr;
}

// virtual
void LGaussianNoise::resize(int batch){
    output->resize(batch);
    noise->resize(batch);
}


void LGaussianNoise::mem_delta() {
    if (this->delta == nullptr){
        // Reserve parent's delta AND assign it to this layer
        parent[0]->mem_delta();

        delta = parent[0]->delta;

        if(this->verbosity_level >= 2){
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}

void LGaussianNoise::free_delta() {
    // Not really needed, but I like to keep all the methods the same (ease the robustness of "copy-paste")
    if(this->delta != nullptr) {
        // Do not delete delta (points to parent)
        delta = nullptr;

        if(this->verbosity_level >= 2){
            std::cout << "Deleted delta for: " + this->name << std::endl;
        }
    }
}

void LGaussianNoise::forward() {
    if (mode == TRMODE) {
        noise->fill_rand_normal_(0.0, stdev);
        Tensor::add(1.0, input, 1.0, noise, output, 0);
    } else {
        Tensor::copy(input, output);
    }
}

void LGaussianNoise::backward() {

}


Layer *LGaussianNoise::share(int c, int bs, vector<Layer *> p) {
    LGaussianNoise *n = new LGaussianNoise(p[0], stdev, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LGaussianNoise::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGaussianNoise *n = new LGaussianNoise(p[0], stdev,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LGaussianNoise::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
