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

    Wix = new Tensor(vector<int>{input->shape[0], units}, dev);
    Wfx = new Tensor(vector<int>{input->shape[0], units}, dev);
    Wox = new Tensor(vector<int>{input->shape[0], units}, dev);
    Wcx = new Tensor(vector<int>{input->shape[0], units}, dev);
    params.push_back(Wix);
    params.push_back(Wfx);
    params.push_back(Wox);
    params.push_back(Wcx);

    gWix = new Tensor(vector<int>{input->shape[0], units}, dev);
    gWfx = new Tensor(vector<int>{input->shape[0], units}, dev);
    gWox = new Tensor(vector<int>{input->shape[0], units}, dev);
    gWcx = new Tensor(vector<int>{input->shape[0], units}, dev);
    gradients.push_back(gWix);
    gradients.push_back(gWfx);
    gradients.push_back(gWox);
    gradients.push_back(gWcx);

    if (parent.size()>1) {
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
    }

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}

// RESIZE , MEM_DELTA states

// virtual
void LLSTM::forward() {

    in=new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(input, 0, Wix, 0, in, 0);
    if (parent.size()>1) {
      Tensor::mult2D(((LLSTM *)parent[1])->state_h, 0, Wih, 0, in, 1);
    }
    Sigmoid(in, in);

    fn=new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(input, 0, Wfx, 0, fn, 0);
    if (parent.size()>1) {
      Tensor::mult2D(((LLSTM *)parent[1])->state_h, 0, Wfh, 0, fn, 1);
    }
    Sigmoid(fn, fn);

    on=new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(input, 0, Wox, 0, on, 0);
    if (parent.size()>1) {
      Tensor::mult2D(((LLSTM *)parent[1])->state_h, 0, Woh, 0, on, 1);
    }
    Sigmoid(on, on);

    cn=new Tensor({input->shape[0], units}, dev);
    Tensor::mult2D(input, 0, Wcx, 0, cn, 0);
    if (parent.size()>1) {
      Tensor::mult2D(((LLSTM *)parent[1])->state_h, 0, Wch, 0, cn, 1);
    }
    Tanh(cn,cn);

    incn=new Tensor({input->shape[0], units}, dev);
    Tensor::el_mult(in,cn,incn,0);

    cn1fn=new Tensor({input->shape[0], units}, dev);
    Tensor::el_mult(((LLSTM *)parent[1])->state_c,fn,cn1fn,0);

    Tensor::add(1.0,incn,1.0,cn1fn,state_c,0);

    Tensor::el_mult(state_c,on,state_h,0);

}

void LLSTM::backward() {
    // TODO: Implement
    //delete in,cn, ....
}


Layer *LLSTM::share(int c, int bs, vector<Layer *> p) {
    LLSTM *n = new LLSTM(p, units, bidirectional, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

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
