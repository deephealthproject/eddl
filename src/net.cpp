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
  // random params
  initialize();

}

/////////////////////////////////////////
void Net::forward()
{
  for(int i=0;i<vfts.size();i++)
    vfts[i]->forward();
}

verr Net::delta(vtensor Y)
{
  verr errors;
  for(int i=0;i<Y.size();i++) {
    errors.push_back(0.0);
    outs[i]->set(0.0);
  }

  for(int i=0;i<lout.size();i++) {
    if (cost[i]=="mse") {
      //delta: (T-Y)
      Tensor::sum(1.0,Y[i],-1.0,lout[i]->output, lout[i]->delta,0);
      // batch error: sum((T-Y)^2)
      Tensor::el_mult(lout[i]->delta,lout[i]->delta,outs[i],0);
      errors[i]=Tensor::total_sum(outs[i]);
   }
   else if (cost[i]=="cent")
    {
      // delta: -t/y + (1-t)/(1-y)

      // batch error: -tlog(y)+(1-t)log(1-y)
      Tensor::cent(Y[i],lout[i]->output,outs[i]);
      errors[i]=Tensor::total_sum(outs[i]);
    }
    //typical case where cent is after a softmax
    else if (cost[i]=="soft_cent")
     {
       // parent->delta: (t-y)
       Tensor::sum(1.0,Y[i],-1.0,lout[i]->output, lout[i]->delta,0);
       lout[i]->delta_bp=1;

       // batch error -tlog(y)+(1-t)log(1-y)
       Tensor::cent(Y[i],lout[i]->output,outs[i]);
       errors[i]=Tensor::total_sum(outs[i]);

     }

  }
  return errors;
}
/////////////////////////////////////////
void Net::backward()
{
  for(int i=0;i<vbts.size();i++)
    vbts[i]->backward();
}

/////////////////////////////////////////
void Net::applygrads(int batch)
{
  optimizer->applygrads(batch);
}

/////////////////////////////////////////
void Net::fit(const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out,int batch, int epochs)
{
  int i,j,k,n;

  if (optimizer==NULL)
    msg("Net is not build ","Net fit");

  vtensor tin=vtensor(in.begin(), in.end());
  vtensor tout=vtensor(out.begin(), out.end());

  // Check list sizes
  if (tin.size()!=lin.size())
    msg("input tensor list does not match with defined input layers","fit");
  if (tout.size()!=lout.size())
      msg("output tensor list does not match with defined output layers","fit");

  // Check data consistency
  n=tin[0]->sizes[0];
  for(i=1;i<tin.size();i++)
   if(tin[i]->sizes[0]!=n)
     msg("different number of samples in input tensor","fit");

  for(i=1;i<tout.size();i++)
    if(tout[i]->sizes[0]!=n)
      msg("different number of samples in output tensor","fit");


  // Create internal variables
  //Input and output batch...
  vtensor X;
  for(i=0;i<tin.size();i++) {
    shape s=tin[i]->getshape();
    s[0]=batch;
    X.push_back(new Tensor(s));
  }
  for(i=0;i<n;i++)
   ind.push_back(i);
  for(i=0;i<batch;i++)
    sind.push_back(0);

  vtensor Y;
  verr errors,err;
  for(i=0;i<tout.size();i++) {
    shape s=tout[i]->getshape();
    s[0]=batch;
    Y.push_back(new Tensor(s));
    errors.push_back(0.0);
    outs.push_back(new Tensor(Y[i]->getshape()));
  }

  // Check sizes w.r.t layers in and out
  for(int i=0;i<lin.size();i++)
    if (!Tensor::eqsize(lin[i]->input,X[i]))
      msg("input tensor shapes does not match","fit");

  for(int i=0;i<lout.size();i++)
    if (!Tensor::eqsize(lout[i]->output,Y[i]))
      msg("output tensor shapes does not match","fit");


  // Start training
  fprintf(stderr,"%d epochs of %d batches of size %d\n",epochs,n/batch,batch);
  for(i=0;i<epochs;i++) {
    for(j=0;j<tout.size();j++) errors[j]=0.0;
    for(j=0;j<2000;j++) {
      // random batches
      for(int k=0;k<batch;k++)
        sind[k]=rand()%n;
      // copy a batch from tin--> X
      for(int k=0;k<lin.size();k++)
        Tensor::select(tin[k],X[k],sind);
      // copy a batch from tout--> Y
      for(int k=0;k<lin.size();k++)
        Tensor::select(tout[k],Y[k],sind);

      err=train_batch(X,Y);

      for(k=0;k<tout.size();k++) errors[k]+=err[k];
      for(k=0;k<tout.size();k++)
        fprintf(stderr,"batch %d errors: %f ",j,errors[k]/(batch*(j+1)));
      fprintf(stderr,"\r");
      //getchar();

    }
  }
}

/////////////////////////////////////////

verr Net::train_batch(vtensor X, vtensor Y)
{
  int i,j;

  // these copies can go from CPU to {CPU,GPU,FPGA}
  for(i=0;i<X.size();i++)
    Tensor::copy(X[i],lin[i]->input);

  verr errors;
  reset(); //gradients=0
  forward();
  errors=delta(Y);
  backward();
  //applygrads(X[0]->sizes[0]);

  return errors;
}




/////////////////////////////////////////
verr Net::train_batch(const initializer_list<Tensor*>& in,const initializer_list<Tensor*>& out)
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

  return train_batch(X,Y);
}














///////////////////////////////////////////












//////
