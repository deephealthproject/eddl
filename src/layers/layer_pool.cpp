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

int pool_created=1;

using namespace std;

// constructors and clones
LPool::LPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p):LPool(parent,ks,st,p,"pool"+to_string(pool_created),DEV_CPU){}
LPool::LPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,string name):LPool(parent,ks,st,p,name,DEV_CPU){}
LPool::LPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,int d):LPool(parent,ks,st,p,"pool"+to_string(pool_created),d){}
LPool::LPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,string name,int d):LPool(parent,new PoolDescriptor(ks,st,p),name,d){}

LPool::LPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p):LPool(parent,ks,st,p,"pool"+to_string(pool_created),DEV_CPU){}
LPool::LPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,string name):LPool(parent,ks,st,p,name,DEV_CPU){}
LPool::LPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,int d):LPool(parent,ks,st,p,"pool"+to_string(pool_created),d){}
LPool::LPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,string name,int d):LPool(parent,new PoolDescriptor(ks,st,p),name,d){}


LPool::LPool(Layer *parent,PoolDescriptor *D,string name, int d):LinLayer(name,d)
{
  if (parent->output->dim!=4) msg("LPool only works over 4D tensors","LPool::LPool");
  pool_created++;

  cd=D;

  input=parent->output;
  cd->build(input);

  output=cd->O;
  delta=cd->D;
  cd->ID=parent->delta;

  parent->addchild(this);
  addparent(parent);

}


Layer *LPool::share(int c,int bs,vector<Layer*>p)
{
  LPool *n=new LPool(p[0],{cd->ksize[0],cd->ksize[1]},{cd->stride[0],cd->stride[1]},{cd->pad[0],cd->pad[1]},"share_"+to_string(c)+name,dev);
  n->orig=this;

  return n;
}

Layer *LPool::clone(int c,int bs,vector<Layer*>p,int todev)
{
  LPool *n=new LPool(p[0],{cd->ksize[0],cd->ksize[1]},{cd->stride[0],cd->stride[1]},{cd->pad[0],cd->pad[1]},"clone_"+to_string(todev)+name,todev);
  n->orig=this;

  return n;
}


string LPool::plot(int c)
{
    string s;

    if (c) s=name+" [label="+"\""+name+"\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s=name+" [label="+"\""+name+"\",style=filled,fontsize=12,fillcolor=red,shape=box]";

    return s;
}



//////////////
// MaxPool2D
//////////////
// constructors and clones
LMPool::LMPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p):LMPool(parent,ks,st,p,"mpool"+to_string(pool_created),DEV_CPU){}
LMPool::LMPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,string name):LMPool(parent,ks,st,p,name,DEV_CPU){}
LMPool::LMPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,int d):LMPool(parent,ks,st,p,"mpool"+to_string(pool_created),d){}
LMPool::LMPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, string p,string name,int d):LMPool(parent,new PoolDescriptor(ks,st,p),name,d){}

LMPool::LMPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p):LMPool(parent,ks,st,p,"mpool"+to_string(pool_created),DEV_CPU){}
LMPool::LMPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,string name):LMPool(parent,ks,st,p,name,DEV_CPU){}
LMPool::LMPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,int d):LMPool(parent,ks,st,p,"mpool"+to_string(pool_created),d){}
LMPool::LMPool(Layer *parent,const initializer_list<int>& ks,const initializer_list<int>& st, const initializer_list<int>& p,string name,int d):LMPool(parent,new PoolDescriptor(ks,st,p),name,d){}

LMPool::LMPool(Layer *parent,PoolDescriptor *D,string name, int d):LPool(parent,D,name,d){}


// virtual
void LMPool::forward()
{
  Tensor::MPool2D(cd);
}

void LMPool::backward()
{

  // backprop delta
  if (parent.size())
    {
      Tensor::MPool2D_back(cd);
    }

}

//////////////
// APool2D
//////////////

//....

//////////////
// SPool2D
//////////////

//....
