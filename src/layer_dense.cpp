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


Dense::Dense(Layer *parent,int dim,string name):Dense(parent,dim,name,DEV_CPU,"FLOAT32"){}
Dense::Dense(Layer *parent,int dim,string name,int d):Dense(parent,dim,name,d,"FLOAT32"){}
Dense::Dense(Layer *parent,int dim,string name,string t):Dense(parent,dim,name,DEV_CPU,t){}

Dense::Dense(Layer *parent,int dim,string name,int d,string t):LinLayer(name,d,t)
{
  if (parent->output->dim!=2) msg("Dense only works over 2D tensors");

  input=parent->output;
  output=new Tensor({input->sizes[0],dim},d,t);
  delta=new Tensor(input->getshape(),d,t);

  W=new Tensor({input->sizes[1],dim},d,t);
  bias=new Tensor({dim},d,t);
  gW=new Tensor({input->sizes[1],dim},d,t);
  gbias=new Tensor({dim},d,t);

  vparams.push_back(W);
  vparams.push_back(bias);

  parent->addchild(this);
  addparent(parent);
}

// virtual
void Dense::forward()
{
  Tensor::mult2D(input,0,W,0,output);
  Tensor::sum2D_rowwise(output,bias,output);
}

void Dense::backward()
{
  if (child!=NULL){
    Tensor::mult2D(input,1,child->delta,0,gW);
    Tensor::mult2D(child->delta,0,W,1,delta);
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
  if (child!=NULL) cout<< "Child layer:"<<child->name<<"\n";
  else cout<<"Child layer: None\n";
  cout<<"Input:\n";
  input->info();
  cout<<"Param:\n";
  W->info();
  cout<<"Output:\n";
  output->info();
  cout<<"===============\n\n";
}
