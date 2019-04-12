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

int conv_created=1;

using namespace std;

// constructors and clones
LConv::LConv(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p):LConv(parent,ks,st,p,"conv"+to_string(conv_created),DEV_CPU){}
LConv::LConv(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,string name):LConv(parent,ks,st,p,name,DEV_CPU){}
LConv::LConv(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,int d):LConv(parent,ks,st,p,"conv"+to_string(conv_created),d){}
LConv::LConv(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,string name,int d):LConv(parent,new ConvolDescriptor(ks,st,p),name,d){}

LConv::LConv(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p):LConv(parent,ks,st,p,"conv"+to_string(conv_created),DEV_CPU){}
LConv::LConv(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,string name):LConv(parent,ks,st,p,name,DEV_CPU){}
LConv::LConv(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,int d):LConv(parent,ks,st,p,"conv"+to_string(conv_created),d){}
LConv::LConv(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,string name,int d):LConv(parent,new ConvolDescriptor(ks,st,p),name,d){}

LConv::LConv(Layer *parent,const vector<int>& ks, const vector<int>& st, string p, int d):LConv(parent,ks,st,p,"conv"+to_string(conv_created),d){}
LConv::LConv(Layer *parent,const vector<int>& ks, const vector<int>& st, string p, string name, int d):LConv(parent,new ConvolDescriptor(ks,st,p),name,d){}

LConv::LConv(Layer *parent,ConvolDescriptor *D,string name, int d):LinLayer(name,d)
{
  if (parent->output->dim!=4) msg("LConv only works over 4D tensors","LConv::LConv");
  conv_created++;

  cd=D;

  input=parent->output;
  cd->build(input);

  output=cd->O;
  delta=cd->D;
  cd->ID=parent->delta;

  params.push_back(cd->K);
  params.push_back(cd->bias);

  gradients.push_back(cd->gK);
  gradients.push_back(cd->gbias);

  parent->addchild(this);
  addparent(parent);

}


// virtual
void LConv::forward()
{
  Tensor::Conv2D(cd);
}

void LConv::backward()
{

  //get gradients with provided delta

  Tensor::Conv2D_grad(cd);
  // backprop delta
  if (parent.size())
    {
      Tensor::Conv2D_back(cd);
    }

}

Layer *LConv::share(int c,int bs,vector<Layer*>p)
{
  LConv *n=new LConv(p[0],{cd->ksize[0],cd->ksize[1],cd->ksize[2]},{cd->stride[0],cd->stride[1]},{cd->pad[0],cd->pad[1]},"share_"+to_string(c)+name,dev);
  n->orig=this;

  //share params
  for(int i=0;i<n->params.size();i++) delete n->params[i];
  n->params.clear();

  n->cd->K=cd->K;
  n->cd->bias=cd->bias;
  new (&n->cd->matK) Eigen::Map<Eigen::MatrixXf>(n->cd->K->ptr,cd->kr*cd->kc*cd->kz,cd->nk);

  n->params.push_back(n->cd->K);
  n->params.push_back(n->cd->bias);

  return n;
}

Layer *LConv::clone(int c,int bs,vector<Layer*>p,int todev)
{
  LConv *n=new LConv(p[0],{cd->ksize[0],cd->ksize[1],cd->ksize[2]},{cd->stride[0],cd->stride[1]},{cd->pad[0],cd->pad[1]},"clone_"+to_string(todev)+name,todev);
  n->orig=this;

  return n;
}


string LConv::plot(int c)
{
    string s;

    if (c) s=name+" [label="+"\""+name+"\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s=name+" [label="+"\""+name+"\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
