
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
#include <stdlib.h>
#include <iostream>

#include "layer_conv.h"

using namespace std;


int LConv::total_layers = 0;

// constructors and clones

LConv::LConv(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st,
             const initializer_list<int> &p, string name, int dev) : LConv(parent, new ConvolDescriptor(ks, st, p), name, dev) {}

LConv::LConv(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding,
int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev) : LConv(parent, new ConvolDescriptor(filters, kernel_size, strides, padding), name, dev) {
    // TODO: Implement (Fix initialization)
};

LConv::LConv(Layer *parent, ConvolDescriptor *D, string name, int dev) : LinLayer(name, dev) {
    if (parent->output->ndim != 4) msg("LConv only works over 4D tensors", "LConv::LConv");

    // Check dev with tensor dev

    // Set default name
    if(name.empty()) this->name = "conv" + to_string(++total_layers);

    cd = D;

    input = parent->output;
    cd->build(input);

    output = cd->O;
    delta = cd->D;
    cd->ID = parent->delta;

    params.push_back(cd->K);
    params.push_back(cd->bias);

    gradients.push_back(cd->gK);
    gradients.push_back(cd->gbias);

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LConv::resize(int batch){

  //cout<<"Resize "<<name<<"\n";

  input = parent[0]->output;
  cd->resize(input);

  output = cd->O;
  delta = cd->D;
  cd->ID = parent[0]->delta;

}

void LConv::forward() {
    Tensor::Conv2D(cd);
}

void LConv::backward() {

    //get gradients with provided delta
    Tensor::Conv2D_grad(cd);
    // backprop delta
    if (parent.size()) {
        Tensor::Conv2D_back(cd);
    }

}

Layer *LConv::share(int c, int bs, vector<Layer *> p) {
    LConv *n = new LConv(p[0], {cd->ksize[0], cd->ksize[1], cd->ksize[2]}, {cd->stride[0], cd->stride[1]},
                         {cd->pad[0], cd->pad[1]}, "share_" + to_string(c) + name, dev);
    n->orig = this;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    n->cd->K = cd->K;
    n->cd->bias = cd->bias;
    new(&n->cd->matK) Eigen::Map<Eigen::MatrixXf>(n->cd->K->ptr, cd->kr * cd->kc * cd->kz, cd->nk);

    n->params.push_back(n->cd->K);
    n->params.push_back(n->cd->bias);

    return n;
}

Layer *LConv::clone(int c, int bs, vector<Layer *> p, int todev) {
    LConv *n = new LConv(p[0], {cd->ksize[0], cd->ksize[1], cd->ksize[2]}, {cd->stride[0], cd->stride[1]},
                         {cd->pad[0], cd->pad[1]}, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LConv::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
