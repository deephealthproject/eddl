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

extern ostream& operator<<(ostream& os, const shape s);

EDDL eddl;


////////////////////////////////////////////////////////
///// EDDL is a wrapper class to ease and define the API
////////////////////////////////////////////////////////

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
  return new LTensor(s,dev);
}

tensor EDDL::T(string fname)
{
  return new LTensor(fname);
}


void EDDL::div(tensor t,float v)
{
  t->input->div(v);
}
//////////////////////////////////////////////////////

layer EDDL::Input(const initializer_list<int>& init){
  return new LInput(new Tensor(init));
}

layer EDDL::Input(const initializer_list<int>& init,int dev){
  return new LInput(new Tensor(init,dev));
}

layer EDDL::Input(tensor t)
{
  return new LInput(t->input);
}

layer EDDL::Input(tensor t,int dev)
{
  return new LInput(t->input);
}
//////////////////////////////////////////////////////
layer EDDL::Dense(layer parent,int dim)
{
  return new LDense(parent,dim,DEV_CPU);
}

layer EDDL::Dense(layer parent,int dim,string name)
{
  return new LDense(parent,dim,name,DEV_CPU);
}
layer EDDL::Dense(layer parent,int dim,int dev)
{
  return new LDense(parent,dim,dev);
}
layer EDDL::Dense(layer parent,int dim,string name,int d)
{
  return new LDense(parent,dim,name,d);
}

//////////////////////////////////////////////////////
layer EDDL::Conv(layer parent,const initializer_list<int>& ks)
{
  return new LConv(parent,ks,{1,1},"same");
}
layer EDDL::Conv(layer parent,const initializer_list<int>& ks,const initializer_list<int>& st)
{
  return new LConv(parent,ks,st,"same");
}
layer EDDL::Conv(layer parent,const initializer_list<int>& ks,const initializer_list<int>& st,string p)
{
  return new LConv(parent,ks,st,p);
}
layer EDDL::Conv(layer parent,const initializer_list<int>& ks,string p)
{
  return new LConv(parent,ks,{1,1},p);
}

//////////////////////////////////////////////////////
layer EDDL::MPool(layer parent,const initializer_list<int>& ks)
{
  return new LMPool(parent,ks,ks,"none");
}
layer EDDL::MPool(layer parent,const initializer_list<int>& ks,const initializer_list<int>& st)
{
  return new LMPool(parent,ks,st,"none");
}
layer EDDL::MPool(layer parent,const initializer_list<int>& ks,const initializer_list<int>& st,string p)
{
  return new LMPool(parent,ks,st,p);
}
layer EDDL::MPool(layer parent,const initializer_list<int>& ks,string p)
{
  return new LMPool(parent,ks,ks,p);
}


//////////////////////////////////////////////////////
layer EDDL::Activation(layer parent,string act)
{
  return new LActivation(parent,act,DEV_CPU);
}

layer EDDL::Activation(layer parent,string act,string name)
{
  return new LActivation(parent,act,name,DEV_CPU);
}
layer EDDL::Activation(layer parent,string act,int dev)
{
  return new LActivation(parent,act,dev);
}
layer EDDL::Activation(layer parent,string act,string name,int d)
{
  return new LActivation(parent,act,name,d);
}

//////////////////////////////////////////////////////
layer EDDL::Reshape(layer parent,const initializer_list<int>& init)
{
  shape s(init.begin(), init.end());
  return new LReshape(parent,s);
}

layer EDDL::Reshape(layer parent,const initializer_list<int>& init,string name)
{
  return new LReshape(parent,init,name,DEV_CPU);
}
layer EDDL::Reshape(layer parent,const initializer_list<int>& init,int dev)
{
  return new LReshape(parent,init,dev);
}
layer EDDL::Reshape(layer parent,const initializer_list<int>& init,string name,int d)
{
  return new LReshape(parent,init,name,d);
}

/////////////////////////////////////////////////////////
layer EDDL::Drop(layer parent, float df)
{
  return new LDrop(parent,df);
}
layer EDDL::Drop(layer parent, float df,string name)
{
  return new LDrop(parent,df,name);
}
layer EDDL::Drop(layer parent, float df,int d)
{
  return new LDrop(parent,df,d);
}
layer EDDL::Drop(layer parent, float df,string name,int d)
{
  return new LDrop(parent,df,name,d);
}

/////////////////////////////////////////////////////////

layer EDDL::Add(const initializer_list<layer>& init)
{

   return new LAdd(vlayer(init.begin(), init.end()));

}
layer EDDL::Add(const initializer_list<layer>& init,string name)
{
  return new LAdd(vlayer(init.begin(), init.end()),name);
}
layer EDDL::Add(const initializer_list<layer>& init,int d)
{
  return new LAdd(vlayer(init.begin(), init.end()),d);
}
layer EDDL::Add(const initializer_list<layer>& init,string name,int d)
{
  return new LAdd(vlayer(init.begin(), init.end()),name,d);
}

////////////////////////////////////////////////////////

layer EDDL::Cat(const initializer_list<layer>& init)
{

   return new LCat(vlayer(init.begin(), init.end()));

}
layer EDDL::Cat(const initializer_list<layer>& init,string name)
{
  return new LCat(vlayer(init.begin(), init.end()),name);
}
layer EDDL::Cat(const initializer_list<layer>& init,int d)
{
  return new LCat(vlayer(init.begin(), init.end()),d);
}
layer EDDL::Cat(const initializer_list<layer>& init,string name,int d)
{
  return new LCat(vlayer(init.begin(), init.end()),name,d);
}

////////////

optimizer EDDL::SGD(const initializer_list<float>& p)
{
  return new sgd(p);
}
void EDDL::change(optimizer o,const initializer_list<float>& p)
{
  o->change(p);
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

void EDDL::plot(model m,string fname)
{
  m->plot(fname);
}

void EDDL::build(model net,optimizer o,const initializer_list<string>& c,const initializer_list<string>& m)
{
  net->build(o,c,m);
}
void EDDL::build(model net,optimizer o,const initializer_list<string>& c,const initializer_list<string>& m,CompServ *cs)
{
  net->build(o,c,m,cs);
}

void EDDL::fit(model net, const initializer_list<LTensor*>& in,const initializer_list<LTensor*>& out,int batch,int epochs)
{
  vltensor ltin=vltensor(in.begin(), in.end());
  vltensor ltout=vltensor(out.begin(), out.end());

  vtensor tin;
  for(int i=0;i<ltin.size();i++)
    tin.push_back(ltin[i]->input);

  vtensor tout;
  for(int i=0;i<ltout.size();i++)
    tout.push_back(ltout[i]->input);


  net->fit(tin,tout,batch,epochs);
}

void EDDL::evaluate(model net, const initializer_list<LTensor*>& in,const initializer_list<LTensor*>& out)
{
  vltensor ltin=vltensor(in.begin(), in.end());
  vltensor ltout=vltensor(out.begin(), out.end());

  vtensor tin;
  for(int i=0;i<ltin.size();i++)
    tin.push_back(ltin[i]->input);

  vtensor tout;
  for(int i=0;i<ltout.size();i++)
    tout.push_back(ltout[i]->input);


  net->evaluate(tin,tout);
}


////

bool exist(string name)
{
  if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
  }
  return false;
}

void EDDL::download_mnist()
{
  string cmd;
  string trX="trX.bin";
  string trY="trY.bin";
  string tsX="tsX.bin";
  string tsY="tsY.bin";

  if ( (!exist(trX)) || (!exist(trY))|| (!exist(tsX)) || (!exist(tsY))) {
    cmd="wget https://www.dropbox.com/s/khrb3th2z6owd9t/trX.bin";
    int status=system(cmd.c_str());
    if (status < 0) {
      msg("wget must be installed","eddl.download_mnist");
      exit(1);
    }

    cmd="wget https://www.dropbox.com/s/m82hmmrg46kcugp/trY.bin";
    status=system(cmd.c_str());
    if (status < 0) {
      msg("wget must be installed","eddl.download_mnist");
      exit(1);
    }
    cmd="wget https://www.dropbox.com/s/7psutd4m4wna2d5/tsX.bin";
    status=system(cmd.c_str());
    if (status < 0) {
      msg("wget must be installed","eddl.download_mnist");
      exit(1);
    }
    cmd="wget https://www.dropbox.com/s/q0tnbjvaenb4tjs/tsY.bin";
    status=system(cmd.c_str());
    if (status < 0) {
      msg("wget must be installed","eddl.download_mnist");
      exit(1);
    }

  }
}












//////
