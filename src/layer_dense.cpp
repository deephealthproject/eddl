// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019 Roberto Paredes Palacios, <rparedes@dsic.upv.es>

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
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;

Dense::Dense(Layer *parent,int dim):Dense(parent,dim,"__dense__",DEV_CPU){}
Dense::Dense(Layer *parent,int dim,string name):Dense(parent,dim,name,DEV_CPU){}
Dense::Dense(Layer *parent,int dim,int dev):Dense(parent,dim,"__dense__",dev){}
Dense::Dense(Layer *parent,int dim,string name,int d):LinLayer(name,d)
{
  if (parent->output->dim!=2) msg("Dense only works over 2D tensors");

  input=parent->output;
  output=new Tensor({input->sizes[0],dim},d);
  delta=new Tensor(input->getshape(),d);

  W=new Tensor({input->sizes[1],dim},d);
  bias=new Tensor({dim},d);
  params.push_back(W);
  params.push_back(bias);

  gW=new Tensor({input->sizes[1],dim},d);
  gbias=new Tensor({dim},d);
  gradients.push_back(gW);
  gradients.push_back(gbias);

  parent->addchild(this);
  addparent(parent);
}

// virtual
void Dense::forward()
{
  Tensor::mult2D(input,0,W,0,output,0);
  Tensor::sum2D_rowwise(output,bias,output);
}

void Dense::backward()
{
  if (child.size()){
    // to deal with multiple childs
    gW->set(0);
    gbias->set(0);
    delta->set(0);
    for(int i = 0; i != child.size(); i++) {
      Tensor::mult2D(input,1,child[i]->delta,0,gW,1);
      Tensor::reduce_sum2D(child->delta,gbias,0,1);
      Tensor::mult2D(child[i]->delta,0,W,1,delta,1);
    }
  }
}

void Dense::applygrads()
{

}

void Dense::info()
{
  cout<<"\n===============\n";
  cout<< "Layer Dense "<<name<<"\n";
  cout<< "Parent layer:"<<parent->name<<"\n";
  cout<< "Child layers:\n";
  if (child.size())
    for(int i = 0; i != child.size(); i++)
      cout<< child[i]->name<<"\n";
  else cout<<"None\n";
  cout<<"Input:\n";
  input->info();
  cout<<"Param:\n";
  W->info();
  cout<<"Output:\n";
  output->info();
  cout<<"===============\n\n";
}
