/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/core/layer_core.h"

extern ostream &operator<<(ostream &os, const vector<int> shape);


using namespace std;

int LReshape::total_layers = 0;

LReshape::LReshape(Layer *parent, vector<int> shape, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    ls = shape;

    if(name.empty()) this->name = "reshape" + to_string(++total_layers);

    input = parent->output;

    vector<int> sin = input->getShape();
    int tin = input->size;
    int t = 1, c = 0, ind = -1;


    // Check shape comp.
    for (int i = 0; i < ls.size(); i++) {
        if (ls[i] != -1) t *= ls[i];
        else {
            if (c) msg("Ambiguous reshape, more than one -1", "Reshape");
            else {
                c = 1;
                ind = i;
            }
        }
    }

    if (c == 1) {

        if (t > tin) {
            for(int i=0;i<sin.size();i++)
              cout<<sin[i]<<" x ";
            cout<<endl;
            msg("Incompatible shape", "Reshape");
        } else if (tin % t) {
          for(int i=0;i<sin.size();i++)
            cout<<sin[i]<<" x ";
            cout<<endl;
            msg("Incompatible shape", "Reshape");
        } else {
            ls[ind] = tin / t;
            t = tin;
        }
    } else if (t != tin) {
      for(int i=0;i<sin.size();i++)
        cout<<sin[i]<<" x ";
        cout<<endl;
        msg("Incompatible shape", "Reshape");
    }

    ///////

    // sharing the pointers to data
#ifdef cFPGA
    printf("creating new tensor output for reshape (at constructor)\n");
#endif

    output = new Tensor(ls, parent->output);

    parent->addchild(this);
    addparent(parent);
}

LReshape::~LReshape(){
    if (output != nullptr) {
        output->ptr = nullptr; // because the memory block of the tensor comes from the parent layer.
        delete output;
    }
    if (delta != nullptr) {
        delta->ptr = nullptr; // because the memory block of the tensor comes from the parent layer.
        delete delta;
    }
    output=delta=nullptr;
}

// virtual
void LReshape::resize(int batch){
    ls[0]=batch;
#ifdef cFPGA
    printf("voy a hacer resize!!!! batch %d shape[0] %d tensor_id %d, tensor_id parent %d fpga_ptr %p\n", batch, output->shape[0], output->fpga_tensor_id, parent[0]->output->fpga_tensor_id, parent[0]->output->fpga_ptr);
    output->resize(batch, parent[0]->output->ptr, parent[0]->output->fpga_ptr, false);
#else
    output->resize(batch, parent[0]->output->ptr, nullptr, false);
#endif
}


void LReshape::mem_delta() {
    if (this->delta == nullptr) {
        // Reserve parent's delta AND assign it to this layer
        parent[0]->mem_delta();

        // Problem: Delta is always created, regardless of the low_mem
#ifdef cFPGA
	printf("creating new delta tensor for reshape at mem_delta\n");
#endif
        delta = new Tensor(ls, parent[0]->delta);

        if(this->verbosity_level >= 2){
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}


void LReshape::free_delta() {
    if(this->delta != nullptr) {
        // Do not delete its delta directly (It's pointer points to parent's delta)
        delta->ptr = nullptr;
#ifdef cFPGA
        delta->fpga_ptr = nullptr;
#endif
        delete delta;
        delta = nullptr;

        if(this->verbosity_level >= 2){
            std::cout << "Deleted delta for: " + this->name << std::endl;
        }
    }
}

void LReshape::forward() {

}


void LReshape::backward() {

}


Layer *LReshape::share(int c, int bs, vector<Layer *> p) {
    vector<int> shape = ls;
    shape[0] = bs;

    auto *n = new LReshape(p[0], shape, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LReshape::clone(int c, int bs, vector<Layer *> p, int todev) {

    vector<int> shape = ls;
    shape[0] = bs;

    auto *n = new LReshape(p[0], shape,  name, todev, mem_level);

    n->orig = this;

    return n;
}


string LReshape::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
