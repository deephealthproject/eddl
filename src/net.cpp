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
  optimizer=NULL;
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
void Net::initialize()
{
  for(int i = 0; i != layers.size(); i++)
    layers[i]->initialize();
}

/////////////////////////////////////////
void Net::reset()
{
  for(int i = 0; i != layers.size(); i++)
    layers[i]->reset();
}

/////////////////////////////////////////
void Net::fts()
{
  int i,j,k,n;
  vector<int> visit;
  vector<int> lin;

  fprintf(stderr,"FTS:");
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
      fprintf(stderr,"%s |",layers[j]->name.c_str());

    visit[j]=1;
    vfts.push_back(layers[j]);

    for(k=0;k<layers[j]->lout;k++)
      for(n=0;n<layers.size();n++)
        if(layers[n]==layers[j]->child[k]) lin[n]--;

  }
  fprintf(stderr,"\n");

}


/////////////////////////////////////////
void Net::bts(){
  int i,j,k,n;
  vector<int> visit;
  vector<int> lout;

 fprintf(stderr,"BTS:");
  for(i=0;i<layers.size();i++) {
    visit.push_back(0);
    lout.push_back(layers[i]->lout);
  }

  for(i=0;i<layers.size();i++) {

    for(j=0;j<layers.size();j++)
      if ((lout[j]==0)&&(!visit[j])) break;

    if (j==layers.size())
      msg("error recurrent net in ","bts");

    if (layers[j]->lin)
      fprintf(stderr,"%s-->",layers[j]->name.c_str());
    else
      fprintf(stderr,"%s |",layers[j]->name.c_str());

    visit[j]=1;
    vbts.push_back(layers[j]);

    for(k=0;k<layers[j]->lin;k++)
      for(n=0;n<layers.size();n++)
        if(layers[n]==layers[j]->parent[k]) lout[n]--;

  }
  fprintf(stderr,"\n");
}


/////////////////////////////////////////
void Net::build(optim *opt,const initializer_list<string>& c)
{

  fprintf(stderr,"Build net\n");
  cost=vstring(c.begin(), c.end());
  if (cost.size()!=lout.size())
    msg("Cost list size does not match output list ","build");

  optimizer=opt;
  optimizer->setlayers(layers);

  // forward sort
  fts();
  // backward sort
  bts();

}

/////////////////////////////////////////
void Net::forward()
{
  for(int i=0;i<vfts.size();i++)
    vfts[i]->forward();
}

void Net::delta(vtensor tout)
{
  for(int i=0;i<lout.size();i++) {
    if (cost[i]=="mse")
     Tensor::sum2D(1.0,tout[i],-1.0,lout[i]->output, lout[i]->delta,0);
  }
}
/////////////////////////////////////////
void Net::backward()
{
  for(int i=0;i<vbts.size();i++)
    vbts[i]->backward();
}

/////////////////////////////////////////
void Net::applygrads()
{
  optimizer->applygrads();
}

/////////////////////////////////////////
void Net::fit(const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out,int batch, int epochs)
{
  int i,j,n;

  if (optimizer==NULL)
    msg("Net is not build ","Net fit");

  vtensor tin=vtensor(in.begin(), in.end());
  vtensor tout=vtensor(out.begin(), out.end());

  // Check sizes
  if (tin.size()!=lin.size())
    msg("input tensor list does not match with defined input layers","fit");
  if (tout.size()!=lout.size())
      msg("output tensor list does not match with defined output layers","fit");

  n=tin[0]->sizes[0];
  for(i=1;i<tin.size();i++)
   if(tin[i]->sizes[0]!=n)
     msg("different number of samples in input tensor","fit");

  for(i=1;i<tout.size();i++)
    if(tout[i]->sizes[0]!=n)
      msg("different number of samples in output tensor","fit");


  // Input and output batch
  vtensor X;
  for(i=0;i<tin.size();i++) {
    shape s=tin[i]->getshape();
    s[0]=batch;
    X.push_back(new Tensor(s));
  }
  vtensor Y;
  for(i=0;i<tout.size();i++) {
    shape s=tout[i]->getshape();
    s[0]=batch;
    Y.push_back(new Tensor(s));
  }

  // Check sizes
  for(int i=0;i<lin.size();i++)
    if (!Tensor::eqsize(lin[i]->input,X[i]))
      msg("input tensor shapes does not match","fit");

  for(int i=0;i<lout.size();i++)
    if (!Tensor::eqsize(lout[i]->output,Y[i]))
      msg("output tensor shapes does not match","fit");


  // Start training
  fprintf(stderr,"%d epochs of %d batches of size %d\n",epochs,n/batch,batch);
  for(j=0;j<epochs;j++) {
    for(i=0;i<2;i++) {
      // copy a batch from tin--> X
      // copy a batch from tout--> Y

      train_batch(X,Y);
    }
  }


}

/////////////////////////////////////////

void Net::train_batch(vtensor tin, vtensor tout)
{
  int i,j;

  // these copies can go from CPU to {CPU,GPU,FPGA}
  for(i=0;i<tin.size();i++)
    Tensor::copy(tin[i],lin[i]->input);

  forward();
  delta(tout);
  backward();
  applygrads();

  fprintf(stderr,"OK train_batch\n");
}




/////////////////////////////////////////

void Net::train_batch(const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out)
{
  int i,j,n;

  vtensor X=vtensor(in.begin(), in.end());
  vtensor Y=vtensor(out.begin(), out.end());

  // Check sizes
  if (X.size()!=lin.size())
    msg("input tensor list does not match","fit");
  if (Y.size()!=lout.size())
      msg("output tensor list does not match","fit");

  for(int i=0;i<lin.size();i++)
    if (!Tensor::eqsize(lin[i]->input,X[i]))
      msg("input tensor shapes does not match","fit");

  for(int i=0;i<lin.size();i++)
    if (!Tensor::eqsize(lout[i]->output,Y[i]))
      msg("output tensor shapes does not match","fit");

  train_batch(X,Y);
}














///////////////////////////////////////////












//////
