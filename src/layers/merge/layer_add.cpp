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

#include "eddl/layers/merge/layer_merge.h"


using namespace std;


int LAdd::total_layers = 0;



LAdd::LAdd(vector<Layer *> parent, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if (parent.size() == 0) msg("Error: LAdd layer with empty list");

    if(name.empty()) this->name = "merge_add" + to_string(++total_layers);

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::sameShape(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LAdd layers with different tensor shape");
            }

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->shape, dev);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LAdd::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


// _profile_fpga_tensor_print(). Prints some values of the tensor
void cpu_tensor_print(Tensor *T) {
  // We read the tensor from FPGA
  printf("tensor print:\n");
  int d1_max = 2;
  int d2_max = 4;
  int d3_max = 4;
  if (T->ndim==4) {
    for (int d0=0; d0<T->shape[0]; d0++) {
    for (int d1=0; d1<d1_max; d1++) {
    for (int d2=0; d2<d2_max; d2++) {
    for (int d3=0; d3<d3_max; d3++) {
    
    //for (int d0=0; d0<T->shape[0]; d0++) {
    //for (int d1=0; d1<T->shape[1]; d1++) {
    //for (int d2=0; d2<T->shape[2]; d2++) {
    //for (int d3=0; d3<T->shape[3]; d3++) {
      int a = (d0 * T->shape[1] * T->shape[2] * T->shape[3]) + (d1 * T->shape[2] * T->shape[3]) + (d2 * T->shape[3]) + d3;
      printf("%f ", T->ptr[a]);
      
    }
    //printf("\n");
    }
    //printf("\n\n");
    }
    //printf("\n\n\n");
    }
  }  else if(T->ndim==2) {
       for (int d0=0; d0<d1_max; d0++) {
       for (int d1=0; d1<d2_max; d1++) {
       //for (int d0=0; d0<T->shape[0]; d0++) {
       //for (int d1=0; d1<T->shape[1]; d1++) {
         int a = (d0 * T->shape[1]) + d1;
         printf("%f ", T->ptr[a]);
       }
       printf("\n\n");
    }

  } else if(T->ndim==1) {
    for (int d0=0; d0<T->shape[0]; d0++) {
      printf("%f ", T->ptr[0]);
    }
    printf("\n\n");
    }
}

void LAdd::forward() {
    printf("\n\n\n PARENT 0\n");
    cpu_tensor_print(parent[0]->output);
    printf("\n\n\n PARENT 1\n");
    cpu_tensor_print(parent[1]->output);
    exit(0);
    output->fill_(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);

}

void LAdd::backward() {
    for (int i = 0; i < parent.size(); ++i) {
        Tensor::inc(delta, parent[i]->delta);
      }
}

Layer *LAdd::share(int c, int bs, vector<Layer *> p) {
    LAdd *n = new LAdd(p, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}


Layer *LAdd::clone(int c, int bs, vector<Layer *> p, int todev) {
    LAdd *n = new LAdd(p,  name, todev,mem_level);
    n->orig = this;

    return n;
}
