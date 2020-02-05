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

LActivation::LActivation(Layer *parent, string act, vector<float> params, string name, int dev, int mem) : LinLayer(name, dev, mem){

    // Set default name
    if(name.empty()) this->name = act + to_string(++total_layers);

    this->act = act;
    this->params = params;

    input = parent->output;
    output = new Tensor(input->shape, dev);
    if (!mem_level){ delta = new Tensor(output->shape, dev); }
    delta_bp = 0;

    parent->addchild(this);
    addparent(parent);
}

// virtual
void LActivation::resize(int batch){
    Layer::resize(batch);
}

void LActivation::forward(){

    if (act == "relu"){
        ReLu(this->input, this->output);

    }else if (act == "elu"){
        float alpha = this->params[0];
        ELu(this->input, this->output, alpha);

    }else if (act == "selu"){
        // https://mlfromscratch.com/activation-functions-explained/#selu
        float alpha = this->params[0];
        float scale = this->params[1];

        ELu(this->input, this->output, alpha);
        this->output->mult_(scale);

    }else if (act == "exp"){
        this->output = Tensor::exp(this->input);

    }else if (act == "softplus"){
        Softplus(this->input, this->output);

    }else if (act == "softsign"){
        Softsign(this->input, this->output);

    }else if (act == "softmax"){
        Softmax(this->input, this->output);

    }else if (act == "sigmoid"){
        Sigmoid(this->input, this->output);

    }else if (act == "hard_sigmoid"){
        HardSigmoid(this->input, this->output);

    }else if (act == "leaky_relu"){
        float alpha = this->params[0];
        LeakyReLu(this->input, this->output, alpha);

    }else if (act == "tanh"){
        Tanh(this->input, this->output);

    }else if (act == "linear"){
        float alpha = this->params[0];
        Linear(this->input, this->output, alpha);
    }
}


void LActivation::backward(){


    if (parent.size()){
        if (parent[0]->mem_level)  parent[0]->mem_delta();
        if (delta_bp){
            Tensor::inc(delta, parent[0]->delta);
        }else {
            if (act == "relu"){
                D_ReLu(delta, input, parent[0]->delta);

            }else if (act == "elu"){
                float alpha = this->params[0];
                D_ELu(delta, input, parent[0]->delta, alpha);

            }else if (act == "selu"){
                // https://mlfromscratch.com/activation-functions-explained/#selu
                float alpha = this->params[0];
                float scale = this->params[1];

                 D_ELu(delta, input, parent[0]->delta, alpha);
                this->output->mult_(scale);

            }else if (act == "exp"){
                // TODO: Review
                Tensor::el_mult(delta, output, parent[0]->delta, 0);

            }else if (act == "softplus"){
                D_softplus(delta, output, parent[0]->delta);

            }else if (act == "softsign"){
                D_softsign(delta, output, parent[0]->delta);

            }else if (act == "softmax"){
                D_Softmax(delta, output, parent[0]->delta);

            }else if (act == "sigmoid"){
                D_Sigmoid(delta, output, parent[0]->delta);

            }else if (act == "hard_sigmoid"){
                D_HardSigmoid(delta, input, parent[0]->delta);

            }else if (act == "leaky_relu"){
                float alpha = this->params[0];
                D_LeakyReLu(delta, input, parent[0]->delta, alpha);

            }else if (act == "tanh"){
                D_Tanh(delta, output, parent[0]->delta);

            }else if (act == "linear"){
                float alpha = this->params[0];
                D_Linear(delta, input, parent[0]->delta, alpha);
            }
        }
      if (mem_level)  free_delta();
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

Layer *LActivation::share(int c, int bs, vector<Layer *> p){
    LActivation *n = new LActivation(p[0], this->act, this->params, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;
    n->delta_bp = delta_bp;

    return n;
}

Layer *LActivation::clone(int c, int bs, vector<Layer *> p, int todev){


    LActivation *n = new LActivation(p[0], this->act, this->params, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;
    n->delta_bp = delta_bp;

    return n;
}


string LActivation::plot(int c){
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightSalmon,shape=box]";

    return s;
}
