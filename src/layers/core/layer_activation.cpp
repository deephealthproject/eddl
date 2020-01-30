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

using namespace std;

int LActivation::total_layers = 0;

LActivation::LActivation(Layer *parent, string act, string name, int dev, float param) : LinLayer(name, dev) {

    // Set default name
    if(name.empty()) this->name = "activation" + to_string(++total_layers);

    this->act = act;
    this->param=param;

    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = new Tensor(output->getShape(), dev);
    delta_bp = 0;

    parent->addchild(this);
    addparent(parent);
}
// virtual
void LActivation::resize(int batch){
    Layer::resize(batch);
}


void LActivation::forward() {

    if (act == "relu") {
        ReLu(this->input, this->output);

    } else if (act == "elu") {
        ELu(this->input, this->output, this->param);

    } else if (act == "selu") {
        // https://mlfromscratch.com/activation-functions-explained/#selu
        float alpha = 1.6732632423543772848170429916717f;
        float scale = 1.0507009873554804934193349852946f;

        ELu(this->input, this->output, alpha);
        this->output->mult_(scale);

    } else if (act == "exp") {
        this->output = Tensor::exp(this->input);

    } else if (act == "softmax") {
        Softmax(this->input, this->output);

    } else if (act == "sigmoid") {
        Sigmoid(this->input, this->output);

    } else if (act == "leaky_relu") {
        LeakyReLu(this->input, this->output, this->param);

    } else if (act == "tanh") {
        Tanh(this->input, this->output);

    }  else if (act == "linear") {
        Linear(this->input, this->output, this->param);
    }
}


void LActivation::backward() {


    if (parent.size()) {
        if (delta_bp) {
            Tensor::inc(delta, parent[0]->delta);
        } else {
            if (act == "relu") {
                D_ReLu(delta, input, parent[0]->delta);

            } else if (act == "elu") {
                D_ELu(delta, input, parent[0]->delta, param);

            } else if (act == "selu") {
                // https://mlfromscratch.com/activation-functions-explained/#selu
                float alpha = 1.6732632423543772848170429916717f;
                float scale = 1.0507009873554804934193349852946f;

                 D_ELu(delta, input, parent[0]->delta, alpha);
                this->output->mult_(scale);

            } else if (act == "exp") {
                // TODO: Review
                Tensor::el_mult(delta, output, parent[0]->delta, 0);

            } else if (act == "softmax") {
                D_Softmax(delta, output, parent[0]->delta);

            } else if (act == "sigmoid") {
                D_Sigmoid(delta, output, parent[0]->delta);

            } else if (act == "leaky_relu") {
                D_LeakyReLu(delta, input, parent[0]->delta,param);

            }  else if (act == "tanh") {
                D_Tanh(delta, output, parent[0]->delta);

            }  else if (act == "linear") {
                D_Linear(delta, input, parent[0]->delta, param);
            }
        }
    }
}


void LActivation::save(std::ofstream &ofs, string format){
    // Save act
    // Save param for "lrelu"
}

void LActivation::load(std::ifstream &ifs, string format){
    // Load act
    // Load param for "lrelu"
}

Layer *LActivation::share(int c, int bs, vector<Layer *> p) {

    LActivation *n = new LActivation(p[0], act, "share_" + to_string(c) + name, dev);
    n->orig = this;
    n->delta_bp = delta_bp;

    return n;
}

Layer *LActivation::clone(int c, int bs, vector<Layer *> p, int todev) {

    LActivation *n = new LActivation(p[0], act, "clone_" + to_string(todev) + name, todev);
    n->orig = this;
    n->delta_bp = delta_bp;

    return n;
}


string LActivation::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + act+ "_" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + act + "_" + name + "\",style=filled,fontsize=12,fillcolor=LightSalmon,shape=box]";

    return s;
}
