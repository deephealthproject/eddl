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

using namespace std;

Add::Add(vector<Layer*> in):Add(in,"__add__",DEV_CPU){}
Add::Add(vector<Layer*> in,int dev):Add(in,"__add__",DEV_CPU){}
Add::Add(vector<Layer*> in,string name):Add(in,name,DEV_CPU){}

Add::Add(vector<Layer*> in,string name,int d):MLayer(name,d)
{
  if (in.size()==0) msg("Error: Add layer with empty list");
  parent=in;
  if (parent.size()>1)
    for(int i=0;i<parent.size()-1;++i)
      if (!Tensor::eqsize(parent[i]->output,parent[i+1]->output))
        msg("Error: Add layers with different tensor sizes");

  input=new Tensor(parent[0]->output->getshape());
  output=new Tensor(parent[0]->output->getshape());

}


// virtual
void Add::info()
{
  cout<<"\n===============\n";
  cout<< "Layer Add "<<name<<"\n";
  cout<< "Layers: "<<name<<"\n";
  for(int i = 0; i != parent.size(); i++)
    {
      cout<< parent[i]->name<<"\n";
      parent[i]->info();
    }

  cout<<"===============\n\n";
}


void Add::forward(){}
void Add::backward(){}
Layer *Add::share(int c,vector<Layer*>p){return NULL;}
Layer *Add::clone(int c,vector<Layer*>p,int todev){return NULL;}





///////////////
