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

extern ostream& operator<<(ostream& os, const shape s);

int reshape_created=1;

using namespace std;

LReshape::LReshape(Layer *parent,const initializer_list<int>& init):LReshape(parent,shape(init.begin(), init.end()),"reshape"+to_string(reshape_created),DEV_CPU){}
LReshape::LReshape(Layer *parent,const initializer_list<int>& init,string name):LReshape(parent,shape(init.begin(), init.end()),name,DEV_CPU){}
LReshape::LReshape(Layer *parent,const initializer_list<int>& init,int dev):LReshape(parent,shape(init.begin(), init.end()),"reshape"+to_string(reshape_created),dev){}
LReshape::LReshape(Layer *parent,const initializer_list<int>& init,string name,int d):LReshape(parent,shape(init.begin(), init.end()),"reshape"+to_string(reshape_created),dev){}

LReshape::LReshape(Layer *parent,shape s):LReshape(parent,s,"reshape"+to_string(reshape_created),DEV_CPU){}
LReshape::LReshape(Layer *parent,shape s,int d):LReshape(parent,s,"reshape"+to_string(reshape_created),d){}
LReshape::LReshape(Layer *parent,shape s,string name):LReshape(parent,s,name,DEV_CPU){}
LReshape::LReshape(Layer *parent,shape s,string name,int d):LinLayer(name,d)
{
  ls=s;
  reshape_created++;

  input=parent->output;

  shape sin=input->getshape();
  int tin=input->tam;
  int t=1,c=0,ind=-1;

  // Check sizes comp.
  for(int i=0;i<ls.size();i++) {
    if (ls[i]!=-1) t*=ls[i];
    else {
      if (c) msg("Ambiguous reshape, more than one -1","Reshape");
      else {c=1;ind=i;}
    }
  }

  if (c==1) {
    if (t>tin){
      msg("Incompatible sizes","Reshape");
    }
    else if (tin%t){
      msg("Incompatible sizes","Reshape");
    }
    else{
      ls[ind]=tin/t;
      t=tin;
    }
  }
  else if (t!=tin){
    msg("Incompatible sizes","Reshape");
  }

  ///////

  // sharing the pointers to data
  output=new Tensor(ls,parent->output);
  delta=new Tensor(ls,parent->delta);

  parent->addchild(this);
  addparent(parent);
}


// virtual
void LReshape::forward()
{

}


void LReshape::backward()
{

}


Layer *LReshape::share(int c,int bs,vector<Layer*>p)
{
  shape s=ls;
  s[0]=bs;

  LReshape *n=new LReshape(p[0],s,"share_"+to_string(c)+name,dev);
  n->orig=this;

  return n;
}

Layer *LReshape::clone(int c,int bs,vector<Layer*>p,int todev)
{
  shape s=ls;
  s[0]=bs;

  LReshape *n=new LReshape(p[0],s,"clone_"+to_string(todev)+name,todev);
  n->orig=this;

  return n;
}


string LReshape::plot(int c)
{
    string s;

    if (c) s=name+" [label="+"\""+name+"\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s=name+" [label="+"\""+name+"\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
