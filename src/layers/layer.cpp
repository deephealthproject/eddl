// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;


////////////////////////////////////
///// BASE LAYER CLASS
////////////////////////////////////

Layer::Layer(string n, int d) {
    mode = TRMODE;
    target = delta = input = output = NULL;
    dev = d;
    name = n;
    lin = lout = 0;
    delta_bp = 0;
}


void Layer::initialize() {
    for (int i = 0; i != params.size(); i++) {
        if (params[i]->ndim == 1)
            params[i]->rand_suniform(0.1);
        else if (params[i]->ndim == 2)
            params[i]->rand_gaussian(0.0, sqrt(2.0 / params[i]->shape[0]));
        else
            params[i]->rand_gaussian(0.0, sqrt(2.0 / (params[i]->size / params[i]->shape[0])));
    }
}


void Layer::reset() {
    delta->set(0.0);
}

void Layer::setmode(int m) {
    mode = m;
}

void Layer::info() {
    cout << "\n===============\n";
    cout << "Layer " << name << "\n";
    if (parent.size()) {
        cout << "Parent layers:\n";
        for (int i = 0; i < parent.size(); i++)
            cout << parent[i]->name << "\n";
    } else cout << "No parent layers\n";

    if (child.size()) {
        cout << "Child layers:\n";
        for (int i = 0; i != child.size(); i++)
            cout << child[i]->name << "\n";
    } else cout << "No child layers\n";

    cout << "Input tensor:\n";
    input->info();

    if (params.size()) {
        cout << "Params:\n";
        for (int i = 0; i < params.size(); i++)
            params[i]->info();
    } else cout << "No params\n";

    cout << "Output tensor:\n";
    output->info();
    cout << "===============\n\n";
}

Tensor Layer::getWeights(){

}

Tensor Layer::setWeights(Tensor bias){

}

Tensor Layer::getBias(){

}

Tensor Layer::setBias(Tensor bias){

}

////////////////////////////////////
///// LINEAR LAYERS
////////////////////////////////////
LinLayer::LinLayer(string n, int d) : Layer(n, d) {}

void LinLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}

void LinLayer::addparent(Layer *l) {
    if (parent.size() != 0) msg("This layers only can have one parent layer", l->name.c_str());
    parent.push_back(l);
    lin++;
}


////////////////////////////////////
///// Multiple LAYERS
////////////////////////////////////
MLayer::MLayer(string n, int d) : Layer(n, d) {}

void MLayer::addchild(Layer *l) {
    child.push_back(l);
    lout++;
}


void MLayer::addparent(Layer *l) {
    parent.push_back(l);
    lin++;
}

