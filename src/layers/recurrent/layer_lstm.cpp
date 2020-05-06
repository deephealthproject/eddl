/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#include <cstdio>
#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/recurrent/layer_recurrent.h"


using namespace std;

int LLSTM::total_layers = 0;

LLSTM::LLSTM(vector<Layer *> parent, int units,  bool bidirectional, string name, int dev, int mem): MLayer(name, dev, mem) {

    this->units = units;
    this->bidirectional = bidirectional;

    isrecurrent=true;

    if (parent[0]->output->ndim != 2) msg("LLSTM only works over 2D tensors", "LLSTM");

    if(name.empty()) this->name = "LSTM" + to_string(++total_layers);

    input = parent[0]->output;
    state_h = output = new Tensor(vector<int>{input->shape[0], units}, dev);
    state_c = new Tensor(vector<int>{input->shape[0], units}, dev);

    states.push_back(state_h);
    states.push_back(state_c);


    Wix = new Tensor(vector<int>{input->shape[1], units}, dev);
    Wfx = new Tensor(vector<int>{input->shape[1], units}, dev);
    Wox = new Tensor(vector<int>{input->shape[1], units}, dev);
    Wcx = new Tensor(vector<int>{input->shape[1], units}, dev);
    params.push_back(Wix);
    params.push_back(Wfx);
    params.push_back(Wox);
    params.push_back(Wcx);

    gWix = new Tensor(vector<int>{input->shape[1], units}, dev);
    gWfx = new Tensor(vector<int>{input->shape[1], units}, dev);
    gWox = new Tensor(vector<int>{input->shape[1], units}, dev);
    gWcx = new Tensor(vector<int>{input->shape[1], units}, dev);
    gradients.push_back(gWix);
    gradients.push_back(gWfx);
    gradients.push_back(gWox);
    gradients.push_back(gWcx);

    Wih = new Tensor(vector<int>{units, units}, dev);
    Wfh = new Tensor(vector<int>{units, units}, dev);
    Woh = new Tensor(vector<int>{units, units}, dev);
    Wch = new Tensor(vector<int>{units, units}, dev);
    params.push_back(Wih);
    params.push_back(Wfh);
    params.push_back(Woh);
    params.push_back(Wch);

    gWih = new Tensor(vector<int>{units, units}, dev);
    gWfh = new Tensor(vector<int>{units, units}, dev);
    gWoh = new Tensor(vector<int>{units, units}, dev);
    gWch = new Tensor(vector<int>{units, units}, dev);
    gradients.push_back(gWih);
    gradients.push_back(gWfh);
    gradients.push_back(gWoh);
    gradients.push_back(gWch);


    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}

// RESIZE , MEM_DELTA states
void LLSTM::mem_delta(){
    // Reserve space for the parent's delta
    if(delta == nullptr){
        delta_h=delta = Tensor::zeros(this->output->shape, this->output->device);
        delta_c = Tensor::zeros(this->output->shape, this->output->device);

        delta_states.push_back(delta_h);
        delta_states.push_back(delta_c);


        if(this->verbosity_level >= 2){
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}
void LLSTM::resize(int batch){
    if (output!=nullptr) {
      output->resize(batch);
      state_c->resize(batch);
    }

}

// virtual
void LLSTM::forward() {
    // input=parent[0]->output
    in=new Tensor({input->shape[0], units}, dev);

    Tensor::mult2D(parent[0]->output, 0, Wix, 0, in, 0);
    if (parent.size()>1) {
      Tensor::mult2D(parent[1]->states[0], 0, Wih, 0, in, 1);
    }
    Sigmoid(in, in);

    fn=new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wfx, 0, fn, 0);
    if (parent.size()>1) {
      Tensor::mult2D(parent[1]->states[0], 0, Wfh, 0, fn, 1);
    }
    Sigmoid(fn, fn);

    on=new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wox, 0, on, 0);
    if (parent.size()>1) {
      Tensor::mult2D(parent[1]->states[0], 0, Woh, 0, on, 1);
    }
    Sigmoid(on, on);

    cn=new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(parent[0]->output, 0, Wcx, 0, cn, 0);
    if (parent.size()>1) {
      Tensor::mult2D(parent[1]->states[0], 0, Wch, 0, cn, 1);
    }
    Tanh(cn,cn);

    incn=new Tensor({input->shape[0], units}, dev);
    Tensor::el_mult(in,cn,incn,0);

    cn1fn=new Tensor({input->shape[0], units}, dev);
    if (parent.size()>1) {
      Tensor::el_mult(parent[1]->states[1],fn,cn1fn,0);
    }
    else {
      cn1fn->fill_(0.0);
    }

    Tensor::add(1.0,incn,1.0,cn1fn,state_c,0);

    sh=new Tensor({input->shape[0], units}, dev);
    Tanh(state_c,sh);

    Tensor::el_mult(sh,on,state_h,0);

}

void LLSTM::backward() {
    //delta_h=delta;

    Tensor *d1=new Tensor(delta->getShape(),dev);
    Tensor *d2=new Tensor(delta->getShape(),dev);

    Tensor::el_mult(delta,on,d1,0);
    Tensor::el_mult(delta,sh,d2,0);

    // output gate
    D_Sigmoid(d2, on, d2);
    Tensor::mult2D(parent[0]->output, 1, d2, 0, gWox, 1);
    if (parent.size()>1)
      Tensor::mult2D(parent[1]->states[0], 1, d2, 0, gWoh, 1);
    Tensor::mult2D(d2, 0, Wox, 1, parent[0]->delta, 1);
    if (parent.size()>1)
        Tensor::mult2D(d2, 0, Woh, 1, parent[1]->delta_states[0], 1);


    D_Tanh(d1, sh, d2);
    Tensor::inc(d2,delta_c);

    // forget gate
    if (parent.size()>1) {
      Tensor::el_mult(delta_c, fn, parent[1]->delta_states[1], 1);
      Tensor::el_mult(delta_c, parent[1]->states[1], d2, 0);


      D_Sigmoid(d2, fn, d2);
      Tensor::mult2D(parent[0]->output, 1, d2, 0, gWfx, 1);
      Tensor::mult2D(parent[1]->states[0], 1, d2, 0, gWfh, 1);

      Tensor::mult2D(d2, 0, Wfx, 1, parent[0]->delta, 1);
      Tensor::mult2D(d2, 0, Wfh, 1, parent[1]->delta_states[0], 1);
    }

    Tensor::el_mult(delta_c, in, d1, 0);
    Tensor::el_mult(delta_c, cn, d2, 0);

    // Input gate
    D_Sigmoid(d2, in, d2);
    Tensor::mult2D(parent[0]->output, 1, d2, 0, gWix, 1);
    if (parent.size()>1)
      Tensor::mult2D(parent[1]->states[0], 1, d2, 0, gWih, 1);
    Tensor::mult2D(d2, 0, Wix, 1, parent[0]->delta, 1);
    if (parent.size()>1)
      Tensor::mult2D(d2, 0, Wih, 1, parent[1]->delta_states[0], 1);

    // Cn
    D_Tanh(d1, cn, d1);
    Tensor::mult2D(parent[0]->output, 1, d1, 0, gWcx, 1);
    if (parent.size()>1)
      Tensor::mult2D(parent[1]->states[0], 1, d1, 0, gWch, 1);

    Tensor::mult2D(d1, 0, Wcx, 1, parent[0]->delta, 1);
    if (parent.size()>1)
      Tensor::mult2D(d1, 0, Wch, 1, parent[1]->delta_states[0], 1);

    delete d1;
    delete d2;
    delete in;
    delete fn;
    delete cn;
    delete on;
    delete incn;
    delete cn1fn;
    delete sh;

}


Layer *LLSTM::share(int c, int bs, vector<Layer *> p) {
    LLSTM *n = new LLSTM(p, units, bidirectional, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    n->Wox = Wox;
    n->Wix = Wix;
    n->Wfx = Wfx;
    n->Wcx = Wcx;
    n->params.push_back(Wix);
    n->params.push_back(Wfx);
    n->params.push_back(Wox);
    n->params.push_back(Wcx);
    if (n->parent.size()>1) {
      n->Woh = Woh;
      n->Wih = Wih;
      n->Wfh = Wfh;
      n->Wch = Wch;
      n->params.push_back(Wih);
      n->params.push_back(Wfh);
      n->params.push_back(Woh);
      n->params.push_back(Wch);
    }


    //share gradients
    for (int i = 0; i < n->gradients.size(); i++) delete n->gradients[i];
    n->gradients.clear();

    n->gWox = gWox;
    n->gWix = gWix;
    n->gWfx = gWfx;
    n->gWcx = gWcx;
    n->gradients.push_back(gWix);
    n->gradients.push_back(gWfx);
    n->gradients.push_back(gWox);
    n->gradients.push_back(gWcx);
    if (n->parent.size()>1) {
      n->gWoh = gWoh;
      n->gWih = gWih;
      n->gWfh = gWfh;
      n->gWch = gWch;
      n->gradients.push_back(gWih);
      n->gradients.push_back(gWfh);
      n->gradients.push_back(gWoh);
      n->gradients.push_back(gWch);
    }



    n->reg=reg;
    n->init=init;


    return n;


}

Layer *LLSTM::clone(int c, int bs, vector<Layer *> p, int todev) {
    LLSTM *n = new LLSTM(p, units, bidirectional,  name, todev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LLSTM::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
