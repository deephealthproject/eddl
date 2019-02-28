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

int dense_created=1;

using namespace std;

LDense::LDense(Layer *parent,int dim):LDense(parent,dim,"dense"+to_string(dense_created),DEV_CPU){}
LDense::LDense(Layer *parent,int dim,string name):LDense(parent,dim,name,DEV_CPU){}
LDense::LDense(Layer *parent,int dim,int dev):LDense(parent,dim,"dense"+to_string(dense_created),dev){}
LDense::LDense(Layer *parent,int dim,string name,int d):LinLayer(name,d)
{
  if (parent->output->dim!=2) msg("LDense only works over 2D tensors","LDense");
  dense_created++;
  this->dim=dim;

  input=parent->output;
  output=new Tensor({input->sizes[0],dim},d);
  delta=new Tensor(output->getshape(),d);

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
void LDense::forward()
{
  Tensor::mult2D(input,0,W,0,output,0);
  Tensor::sum2D_rowwise(output,bias,output);
}


void LDense::backward()
{

  //get gradients with provided delta
  Tensor::mult2D(input,1,delta,0,gW,0);
  Tensor::reduce_sum2D(delta,gbias,0,0);
  // backprop delta
  if (parent.size())
    {
      //1: note that increment parent delta
      Tensor::mult2D(delta,0,W,1,parent[0]->delta,1);
    }

}


Layer *LDense::share(int c,int bs,vector<Layer*>p)
{
  LDense *n=new LDense(p[0],dim,"share_"+to_string(c)+name,dev);
  n->orig=this;

  //share params
  for(int i=0;i<n->params.size();i++) delete n->params[i];
  n->params.clear();

  n->W=params[0];
  n->bias=params[1];
  n->params.push_back(n->W);
  n->params.push_back(n->bias);

  return n;
}

Layer *LDense::clone(int c,int bs,vector<Layer*>p,int todev)
{
  LDense *n=new LDense(p[0],dim,"clone_"+to_string(todev)+name,todev);
  n->orig=this;

  return n;
}



string LDense::plot(int c)
{
    string s;

    if (c) s=name+" [label="+"\""+name+"\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s=name+" [label="+"\""+name+"\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
