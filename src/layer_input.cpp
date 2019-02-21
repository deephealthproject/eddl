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

int input_created=1;

LInput::LInput(Tensor *in):LInput(in,"input"+to_string(input_created),DEV_CPU){}
LInput::LInput(Tensor *in,int d):LInput(in,"input"+to_string(input_created),d){}
LInput::LInput(Tensor *in,string name):LInput(in,name,DEV_CPU){}
LInput::LInput(Tensor *in,string name,int d):LinLayer(name,d)
{
  input_created++;
  input=output=in;
  delta=new Tensor(input->getshape(),d);
}


// virtual
void LInput::info()
{
  cout<<"\n===============\n";
  cout<< "Layer LInput "<<name<<"\n";
  input->info();
  cout<<"===============\n\n";
}


void LInput::forward()
{
  delta->set(0.0);
}


void LInput::backward()
{
}


Layer *LInput::share(int c,int bs,vector<Layer*>p)
{
  shape s=input->getshape();
  s[0]=bs;

  LInput *n=new LInput(new Tensor(s),"share_"+to_string(c)+name,dev);
  n->orig=this;

  return n;
}

Layer *LInput::clone(int c,int bs,vector<Layer*>p,int todev)
{
  shape s=input->getshape();
  s[0]=bs;

  LInput *n=new LInput(new Tensor(s,todev),"clone_"+to_string(todev)+name,todev);
  n->orig=this;

  return n;
}



//////
