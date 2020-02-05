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

#include "layer_noise.h"


using namespace std;

int LGaussianNoise::total_layers = 0;

LGaussianNoise::LGaussianNoise(Layer *parent, float stdev, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "gaussiannoise" + to_string(++total_layers);
    this->stdev = stdev;

    // TODO: Implement
    input = parent->output;
    output = new Tensor(input->shape, dev);
    if (!mem_level) delta = parent->delta;
    noise = new Tensor(input->shape, dev);

    parent->addchild(this);
    addparent(parent);

}


LGaussianNoise::~LGaussianNoise()
{
  delete noise;
  delta=nullptr; // is destroyed by parent

}

// virtual
void LGaussianNoise::resize(int batch){
  output->resize(batch);
  noise->resize(batch);
  if (target!=nullptr) target->resize(batch);
}

void LGaussianNoise::forward() {
    // Reserve parent's delta
    if (parent[0]->mem_level) {
        parent[0]->mem_delta();
        delta=parent[0]->delta;
    }

  if (mode == TRMODE) {
      noise->rand_normal(0.0, stdev);
      Tensor::add(1.0, input, 1.0, noise, output, 0);
  } else {
      Tensor::copy(input, output);
  }
}

void LGaussianNoise::backward() {

}


Layer *LGaussianNoise::share(int c, int bs, vector<Layer *> p) {
    LGaussianNoise *n = new LGaussianNoise(p[0], stdev, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LGaussianNoise::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGaussianNoise *n = new LGaussianNoise(p[0], stdev, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LGaussianNoise::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
