/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/core/layer_core.h"

using namespace std;


int LEmbedding::total_layers = 0;

LEmbedding::LEmbedding(Layer *parent, int vocsize, int length, int dim, string name, int dev, int mem): LinLayer(name, dev, mem) {
    // TODO: Implement
    if(name.empty()) this->name = "embedding" + to_string(++total_layers);


    this->length=length;
    this->vocsize=vocsize;
    this->dim=dim;


    input = parent->output;

    // Embedding layers allow input tensors with the following shapes:
    // {batch,length} dim=2
    // {batch} dim=1 but then length must be 1

    if (length>1) {
      if (input->ndim!=2)
        msg("Embeding Layers accepts 2D tensors: {batch,length}","LEmbedding");
      else if (length!=input->shape[1]) {
        cout<<input->shape[1]<<"!="<<length<<endl;
        msg("Lengths don't match input tensor!=embedding length ","LEmbedding");
      }
    }
    else {
      if (input->ndim!=1) {
        if (input->ndim!=2)
          msg("Embeding Layers accepts 2D tensors: {batch,length}","LEmbedding");
        else if (length!=input->shape[1]) // should be 1
          msg("Lengths don't match input tensor!=embedding length ","LEmbedding");
      }
    }

    input = parent->output;
    output = new Tensor(vector<int>{input->shape[0], length*dim}, dev);

    E=new Tensor({vocsize,dim},dev);
    params.push_back(E);
    gE=new Tensor({vocsize,dim},dev);
    gradients.push_back(gE);


    parent->addchild(this);
    addparent(parent);

}

void LEmbedding::forward()
{

  int b=input->shape[0];
  int indim=input->ndim;

  input->reshape_({b*length});

  sind.clear();

  Tensor *inputc=input->clone();
  inputc->toCPU();

  for(int i=0;i<b*length;i++) {
    sind.push_back(inputc->get_({i}));
    if (sind[i]>=vocsize) msg("word > vocsize","LEmbedding::forward");
  }
  delete inputc;

  output->reshape_({b*length,dim});


  Tensor::select(E,output, sind, 0,sind.size()); //1=inc

 /*
  for(int i=0;i<sind.size();i++) {
    Tensor *t2= E->select({to_string(sind[i]), ":"});
    output->set_select({to_string(i),":"},t2);
    delete t2;
  }
*/
  if (indim==2) input->reshape_({b,length});

  output->reshape_({b,length*dim});
}


void LEmbedding::backward()
{
   int b=output->shape[0];

   delta->reshape_({b*length,dim});

   Tensor::deselect(delta,gE, sind, 0,sind.size(),1); //1=inc

   delta->reshape_({b,length*dim});

   if (trainable) if(reg!= nullptr) {reg->apply(E);}

   //cout<<"\n"<<"E="<<E->sum()<<"\n";

}




Layer *LEmbedding::share(int c, int bs, vector<Layer *> p) {
    LEmbedding *n = new LEmbedding(p[0],vocsize, length, dim,  "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;

    //share params
    delete n->params[0];
    delete n->gradients[0];
    n->params.clear();
    n->gradients.clear();

    n->E = params[0];
    n->gE= gradients[0];

    n->params.push_back(n->E);
    n->gradients.push_back(n->gE);

    n->reg=reg;
    n->init=init;
    return n;
}

Layer *LEmbedding::clone(int c, int bs, vector<Layer *> p, int todev) {
    LEmbedding *n = new LEmbedding(p[0],vocsize, length, dim, "share_"+to_string(c)+this->name, todev, this->mem_level);
    n->orig = this;

    n->reg=reg;
    n->init=init;

    return n;
}


string LEmbedding::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}



//////
