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

#include "net.h"

using namespace std;

ostream& operator<<(ostream& os, const shape s) {
  int i;
  os << "(";
  for (i = 0; i < s.size()-1; ++i) {
      os << s[i];
      os <<"x";
  }
  os<<s[i]<<")";

  return os;
}


////////////////////////////////////
///// BASE NET CLASS
////////////////////////////////////

Net::Net(const initializer_list<Layer*>& in,const initializer_list<Layer*>& out):Net(vlayer(in.begin(), in.end()),vlayer(out.begin(), out.end())){}

  /////////////////////////////////////////

Net::Net(vlayer in,vlayer out)
{
  lin=in;
  lout=out;
  for(int i=0;i<lin.size();i++)
    walk(lin[i]);
}
/////////////////////////////////////////
int Net::inNet(Layer *l)
{
  for(int i = 0; i != layers.size(); i++)
    if (l==layers[i]) return 1;
  return 0;
}
/////////////////////////////////////////
  void Net::walk(Layer *l)
{
  if (!inNet(l)) {
    layers.push_back(l);
    for(int i = 0; i != l->child.size(); i++)
      walk(l->child[i]);
    }
}

void Net::info(){

  for(int i=0;i<layers.size();i++)
    fprintf(stderr,"%s ",layers[i]->name.c_str());

  cout<<"\n";
  for(int i=0;i<layers.size();i++) {
    cout<<layers[i]->name<<": ";
    shape si=layers[i]->input->getshape();
    shape so=layers[i]->output->getshape();
    cout<<si<<"-->"<<so<<"\n";
  }

  fprintf(stderr,"\n");

}
/////////////////////////////////////////
void Net::initialize(){}

/////////////////////////////////////////
void Net::reset(){}

/////////////////////////////////////////
void Net::fts()
{
  int i,j,k,n;
  vector<int> visit;
  vector<int> lin;

  for(i=0;i<layers.size();i++) {
    visit.push_back(0);
    lin.push_back(layers[i]->lin);
  }

  for(i=0;i<layers.size();i++) {

    for(j=0;j<layers.size();j++)
      if ((lin[j]==0)&&(!visit[j])) break;

    if (j==layers.size())
      msg("error recurrent net in ","fts");

    if (layers[j]->lout)
      fprintf(stderr,"%s-->",layers[j]->name.c_str());
    else
      fprintf(stderr,"%s||",layers[j]->name.c_str());

    visit[j]=1;
    vfts.push_back(layers[j]);

    for(k=0;k<layers[j]->lout;k++)
      for(n=0;n<layers.size();n++)
        if(layers[n]==layers[j]->child[k]) lin[n]--;

  }
  fprintf(stderr,"\n");

}

/////////////////////////////////////////
void Net::forward(){}

/////////////////////////////////////////
void Net::bts(){}

/////////////////////////////////////////
void Net::backward(){}

/////////////////////////////////////////
void Net::applygrads(){}

/////////////////////////////////////////
void Net::fit(const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out,int batch)
{
  int i,j,n;

  vtensor tin=vtensor(in.begin(), in.end());
  vtensor tout=vtensor(out.begin(), out.end());

  // Check sizes
  if (tin.size()!=lin.size())
    msg("input tensor list does not match","fit");
  if (tout.size()!=lout.size())
      msg("output tensor list does not match","fit");

  n=tin[0]->sizes[0];
  for(i=1;i<tin.size();i++)
   if(tin[i]->sizes[0]!=n)
     msg("different number os samples","fit");

  for(i=1;i<tout.size();i++)
    if(tout[i]->sizes[0]!=n)
      msg("different number os samples","fit");

  fprintf(stderr,"%d batches of size %d\n",n/batch,batch);
  fts();

}

/////////////////////////////////////////
void Net::train_batch(const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out)
{
  vtensor tin=vtensor(in.begin(), in.end());
  vtensor tout=vtensor(out.begin(), out.end());

  // Check sizes
  if (tin.size()!=lin.size())
    msg("input tensor list does not match","fit");
  if (tout.size()!=lout.size())
      msg("output tensor list does not match","fit");

  for(int i=0;i<lin.size();i++)
    if (!Tensor::eqsize(lin[i]->input,tin[i]))
      msg("input tensor shapes does not match","fit");


  for(int i=0;i<lin.size();i++)
    if (!Tensor::eqsize(lout[i]->output,tout[i]))
      msg("output tensor shapes does not match","fit");

  fprintf(stderr,"OK train_batch\n");
}




















///////////////////////////////////////////












//////
