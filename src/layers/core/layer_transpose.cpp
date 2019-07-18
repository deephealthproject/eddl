
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_core.h"

using namespace std;

int LTranspose::total_layers = 0;

LTranspose::LTranspose(Layer *parent, const initializer_list<int> &dims, string name, int dev):LTranspose(parent, vector<int>(dims.begin(), dims.end()), name,dev){}

LTranspose::LTranspose(Layer *parent, vector<int> dims, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "transpose" + to_string(++total_layers);
    this->dims = dims;

    input=parent->output;
    output=new Tensor(input->getShape(),dev);
    delta=new Tensor(input->getShape(),dev);

    parent->addchild(this);
    addparent(parent);
}

void LTranspose::resize(int batch){
  Layer::resize(batch);
}

void LTranspose::forward() {
   Tensor::transpose(input,output,dims);
}


void LTranspose::backward() {
   //Tensor::transpose(delta,delta,rdims);
   //Tensor::inc(delta,parent[0]->delta);
}


string LTranspose::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}

Layer *LTranspose::clone(int c, int bs, vector<Layer *> p, int todev) {
  LTranspose *n;
  n = new LTranspose(p[0], dims, "share_" + to_string(c) + name, todev);
  n->orig = this;
  return n;
}

Layer *LTranspose::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}
