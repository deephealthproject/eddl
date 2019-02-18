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
#include <stdlib.h>
#include <iostream>


#include "eddl.h"


using namespace std;


EDDL eddl;


tensor EDDL::T(const initializer_list<int>& init){
  return T(shape(init.begin(), init.end()),DEV_CPU);
}

tensor EDDL::T(const initializer_list<int>& init, int dev){
  return T(shape(init.begin(), init.end()),dev);
}

tensor EDDL::T(const shape s){
  return T(s,DEV_CPU);
}

tensor EDDL::T(const shape s, int dev) {
  return new Tensor(s,dev);
}

tensor EDDL::T(string fname)
{
  return new Tensor(fname);
}

void EDDL::div(tensor t,float v)
{
  t->div(v);
}
//////////////////////////////////////////////////////

layer EDDL::Input(const initializer_list<int>& init){
  return new Input::Input(new Tensor(init));
}

layer EDDL::Input(const initializer_list<int>& init,int dev){
  return new Input::Input(new Tensor(init,dev));
}

layer EDDL::Input(tensor t)
{
  return new Input::Input(t);
}

layer EDDL::Input(tensor t,int dev)
{
  return new Input::Input(t);
}
//////////////////////////////////////////////////////
layer EDDL::Dense(layer parent,int dim)
{
  return new Dense::Dense(parent,dim,DEV_CPU);
}

layer EDDL::Dense(layer parent,int dim,string name)
{
  return new Dense::Dense(parent,dim,name,DEV_CPU);
}
layer EDDL::Dense(layer parent,int dim,int dev)
{
  return new Dense::Dense(parent,dim,dev);
}
layer EDDL::Dense(layer parent,int dim,string name,int d)
{
  return new Dense::Dense(parent,dim,name,d);
}

//////////////////////////////////////////////////////
layer EDDL::Activation(layer parent,string act)
{
  return new Activation::Activation(parent,act,DEV_CPU);
}

layer EDDL::Activation(layer parent,string act,string name)
{
  return new Activation::Activation(parent,act,name,DEV_CPU);
}
layer EDDL::Activation(layer parent,string act,int dev)
{
  return new Activation::Activation(parent,act,dev);
}
layer EDDL::Activation(layer parent,string act,string name,int d)
{
  return new Activation::Activation(parent,act,name,d);
}

/////////////////////////////////////////////////////////
model EDDL::Model(vlayer in,vlayer out)
{
  return new Net(in,out);
}

////////////

void EDDL::info(model m)
{
  m->info();
}
void EDDL::build(model net,optim *opt,const initializer_list<string>& c,const initializer_list<string>& m)
{
  net->build(opt,c,m);
}
void EDDL::build(model net,optim *opt,const initializer_list<string>& c,const initializer_list<string>& m,int todev)
{
  net->build(opt,c,m,todev);
}

void EDDL::fit(model net, const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out,int batch,int epochs)
{
  net->fit(in,out,batch,epochs);
}
















//////
